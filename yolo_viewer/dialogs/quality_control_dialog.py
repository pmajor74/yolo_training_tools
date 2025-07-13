"""Quality control dialog for auto-annotation assessment."""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import numpy as np
from collections import defaultdict, Counter

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
        total_steps = 12  # Increased for new metrics
        current_step = 0
        
        # 1. Coverage metric
        current_step += 1
        self.progress.emit(current_step, total_steps, "Calculating coverage...")
        coverage = self._calculate_coverage()
        metrics.append(coverage)
        self.metricCalculated.emit(coverage)
        
        # 2. Confidence distribution
        current_step += 1
        self.progress.emit(current_step, total_steps, "Analyzing confidence distribution...")
        conf_dist = self._calculate_confidence_distribution()
        metrics.append(conf_dist)
        self.metricCalculated.emit(conf_dist)
        
        # 3. Class balance
        current_step += 1
        self.progress.emit(current_step, total_steps, "Checking class balance...")
        class_balance = self._calculate_class_balance()
        metrics.append(class_balance)
        self.metricCalculated.emit(class_balance)
        
        # 4. Annotation density
        current_step += 1
        self.progress.emit(current_step, total_steps, "Measuring annotation density...")
        density = self._calculate_annotation_density()
        metrics.append(density)
        self.metricCalculated.emit(density)
        
        # 5. Modification rate
        current_step += 1
        self.progress.emit(current_step, total_steps, "Analyzing modifications...")
        mod_rate = self._calculate_modification_rate()
        metrics.append(mod_rate)
        self.metricCalculated.emit(mod_rate)
        
        # 6. Uncertainty sampling metric
        current_step += 1
        self.progress.emit(current_step, total_steps, "Analyzing uncertainty regions...")
        uncertainty = self._calculate_uncertainty_metric()
        metrics.append(uncertainty)
        self.metricCalculated.emit(uncertainty)
        
        # 7. Edge case detection
        current_step += 1
        self.progress.emit(current_step, total_steps, "Detecting edge cases...")
        edge_cases = self._calculate_edge_cases()
        metrics.append(edge_cases)
        self.metricCalculated.emit(edge_cases)
        
        # 8. Class-specific confidence
        current_step += 1
        self.progress.emit(current_step, total_steps, "Analyzing per-class confidence...")
        class_conf = self._calculate_class_confidence()
        metrics.append(class_conf)
        self.metricCalculated.emit(class_conf)
        
        # 9. Training readiness
        current_step += 1
        self.progress.emit(current_step, total_steps, "Assessing training readiness...")
        readiness = self._calculate_training_readiness()
        metrics.append(readiness)
        self.metricCalculated.emit(readiness)
        
        # 10. Aspect ratio distribution
        current_step += 1
        self.progress.emit(current_step, total_steps, "Analyzing box dimensions...")
        aspect_ratio = self._calculate_aspect_ratio_distribution()
        metrics.append(aspect_ratio)
        self.metricCalculated.emit(aspect_ratio)
        
        # 11. Efficiency metric
        current_step += 1
        self.progress.emit(current_step, total_steps, "Calculating efficiency...")
        efficiency = self._calculate_efficiency_metric()
        metrics.append(efficiency)
        self.metricCalculated.emit(efficiency)
        
        # 12. Data diversity score
        current_step += 1
        self.progress.emit(current_step, total_steps, "Measuring data diversity...")
        diversity = self._calculate_diversity_score()
        metrics.append(diversity)
        self.metricCalculated.emit(diversity)
        
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
    
    def _calculate_uncertainty_metric(self) -> QualityMetric:
        """Calculate uncertainty sampling score - images with mid-range confidence."""
        uncertainty_count = 0
        total_proposals = 0
        
        for proposals in self.session.proposals.values():
            for prop in proposals:
                total_proposals += 1
                if 0.3 <= prop.confidence <= 0.7:
                    uncertainty_count += 1
        
        if total_proposals == 0:
            score = 0
        else:
            score = uncertainty_count / total_proposals
            
        if score >= 0.2:
            status = QualityStatus.WARNING
            desc = f"{uncertainty_count} uncertain predictions - good for active learning"
        elif score >= 0.1:
            status = QualityStatus.GOOD
            desc = "Some uncertain predictions available"
        else:
            status = QualityStatus.GOOD
            desc = "Model is confident - may need harder examples"
            
        return QualityMetric("Uncertainty Score", score, status, desc)
    
    def _calculate_edge_cases(self) -> QualityMetric:
        """Detect edge cases based on unusual annotation patterns."""
        edge_case_count = 0
        images_checked = 0
        
        for img_path, proposals in self.session.proposals.items():
            images_checked += 1
            
            # Check for edge cases
            if len(proposals) == 0:  # No detections
                edge_case_count += 1
            elif len(proposals) > 20:  # Very dense annotations
                edge_case_count += 1
            else:
                # Check for unusual box sizes
                for prop in proposals:
                    x, y, w, h = prop.bbox
                    # Very small or very large boxes
                    if w * h < 400 or w * h > 100000:  # < 20x20 or > 316x316 pixels
                        edge_case_count += 1
                        break
        
        if images_checked == 0:
            score = 0
        else:
            score = edge_case_count / images_checked
            
        if score >= 0.2:
            status = QualityStatus.WARNING
            desc = f"{edge_case_count} edge cases found - review needed"
        elif score >= 0.1:
            status = QualityStatus.WARNING
            desc = "Some edge cases detected"
        else:
            status = QualityStatus.GOOD
            desc = "Normal annotation patterns"
            
        return QualityMetric("Edge Cases", score, status, desc)
    
    def _calculate_class_confidence(self) -> QualityMetric:
        """Calculate per-class confidence levels."""
        class_confidences = defaultdict(list)
        
        for proposals in self.session.proposals.values():
            for prop in proposals:
                if prop.is_approved:
                    class_confidences[prop.class_id].append(prop.confidence)
        
        if not class_confidences:
            return QualityMetric("Class Confidence", 0, QualityStatus.ERROR, 
                               "No approved annotations")
        
        # Find classes with low average confidence
        low_conf_classes = 0
        total_classes = len(class_confidences)
        
        for class_id, confidences in class_confidences.items():
            avg_conf = np.mean(confidences)
            if avg_conf < 0.5:
                low_conf_classes += 1
        
        if total_classes == 0:
            score = 0
        else:
            score = 1 - (low_conf_classes / total_classes)
            
        if score >= 0.9:
            status = QualityStatus.GOOD
            desc = "All classes detected confidently"
        elif score >= 0.7:
            status = QualityStatus.WARNING
            desc = f"{low_conf_classes} classes have low confidence"
        else:
            status = QualityStatus.ERROR
            desc = "Many classes poorly detected"
            
        return QualityMetric("Class Confidence", score, status, desc)
    
    def _calculate_training_readiness(self) -> QualityMetric:
        """Assess if dataset is ready for training."""
        class_counts = defaultdict(int)
        
        for proposals in self.session.proposals.values():
            for prop in proposals:
                if prop.is_approved:
                    class_counts[prop.class_id] += 1
        
        if not class_counts:
            return QualityMetric("Training Readiness", 0, QualityStatus.ERROR,
                               "No annotations available")
        
        # Check minimum samples per class
        classes_below_threshold = 0
        for count in class_counts.values():
            if count < 100:
                classes_below_threshold += 1
        
        total_classes = len(class_counts)
        if total_classes == 0:
            score = 0
        else:
            score = 1 - (classes_below_threshold / total_classes)
            
        if score >= 0.9:
            status = QualityStatus.GOOD
            desc = "Dataset ready for training"
        elif score >= 0.7:
            status = QualityStatus.WARNING
            desc = f"{classes_below_threshold} classes need more samples"
        else:
            status = QualityStatus.ERROR
            desc = "Insufficient data for training"
            
        return QualityMetric("Training Readiness", score, status, desc)
    
    def _calculate_aspect_ratio_distribution(self) -> QualityMetric:
        """Analyze aspect ratio distribution of bounding boxes."""
        aspect_ratios = []
        
        for proposals in self.session.proposals.values():
            for prop in proposals:
                if prop.is_approved:
                    x, y, w, h = prop.bbox
                    if h > 0:
                        aspect_ratios.append(w / h)
        
        if not aspect_ratios:
            return QualityMetric("Aspect Ratio Diversity", 0, QualityStatus.ERROR,
                               "No boxes to analyze")
        
        # Calculate diversity using coefficient of variation
        cv = np.std(aspect_ratios) / np.mean(aspect_ratios) if np.mean(aspect_ratios) > 0 else 0
        
        if cv >= 0.5:
            status = QualityStatus.GOOD
            desc = "Good variety of box shapes"
        elif cv >= 0.3:
            status = QualityStatus.WARNING
            desc = "Moderate shape diversity"
        else:
            status = QualityStatus.WARNING
            desc = "Limited box shape variety"
            
        return QualityMetric("Aspect Ratio Diversity", cv, status, desc)
    
    def _calculate_efficiency_metric(self) -> QualityMetric:
        """Calculate annotation efficiency metrics."""
        stats = self.session.get_statistics()
        
        # Assume average manual annotation time of 30 seconds per image
        manual_time_per_image = 30  # seconds
        auto_time_per_image = 2  # seconds (inference + review)
        
        images_processed = stats['processed_images']
        time_saved = images_processed * (manual_time_per_image - auto_time_per_image)
        
        # Convert to hours
        hours_saved = time_saved / 3600
        
        # Calculate efficiency ratio
        if images_processed > 0:
            efficiency = (manual_time_per_image - auto_time_per_image) / manual_time_per_image
        else:
            efficiency = 0
            
        status = QualityStatus.GOOD
        desc = f"~{hours_saved:.1f} hours saved vs manual annotation"
        
        return QualityMetric("Time Efficiency", efficiency, status, desc)
    
    def _calculate_diversity_score(self) -> QualityMetric:
        """Calculate dataset diversity score."""
        # Analyze various diversity factors
        images_with_annotations = 0
        annotation_counts = []
        class_distribution = defaultdict(int)
        
        for proposals in self.session.proposals.values():
            if proposals:
                images_with_annotations += 1
                annotation_counts.append(len(proposals))
                for prop in proposals:
                    if prop.is_approved:
                        class_distribution[prop.class_id] += 1
        
        if images_with_annotations == 0:
            return QualityMetric("Data Diversity", 0, QualityStatus.ERROR,
                               "No annotated images")
        
        # Calculate diversity factors
        factors = []
        
        # 1. Annotation count variance
        if annotation_counts:
            cv_counts = np.std(annotation_counts) / np.mean(annotation_counts) if np.mean(annotation_counts) > 0 else 0
            factors.append(min(cv_counts, 1.0))
        
        # 2. Class distribution entropy
        if class_distribution:
            total = sum(class_distribution.values())
            probs = [count/total for count in class_distribution.values()]
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
            normalized_entropy = entropy / np.log(len(class_distribution)) if len(class_distribution) > 1 else 0
            factors.append(normalized_entropy)
        
        # 3. Coverage factor
        coverage = images_with_annotations / self.session.total_images if self.session.total_images > 0 else 0
        factors.append(coverage)
        
        # Average diversity score
        diversity_score = np.mean(factors) if factors else 0
        
        if diversity_score >= 0.7:
            status = QualityStatus.GOOD
            desc = "High dataset diversity"
        elif diversity_score >= 0.5:
            status = QualityStatus.WARNING
            desc = "Moderate diversity - consider adding varied examples"
        else:
            status = QualityStatus.ERROR
            desc = "Low diversity - add more varied data"
            
        return QualityMetric("Data Diversity", diversity_score, status, desc)


class QualityControlDialog(QDialog):
    """Dialog for quality control assessment."""
    
    def __init__(self, session: AutoAnnotationSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.metrics: List[QualityMetric] = []
        self.class_names = {}  # Will be populated from parent if available
        self.setWindowTitle("Quality Control Assessment")
        self.setModal(True)
        self.resize(900, 700)
        self._setup_ui()
        
        # Try to get class names from parent
        if parent and hasattr(parent, '_class_names'):
            self.class_names = parent._class_names
        
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
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality_score()
        
        # Predict model performance
        predicted_map = self._predict_model_performance()
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                td, th {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                .good {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
                .metric-group {{ margin-bottom: 30px; }}
                .recommendation {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #0066cc; }}
                .priority-high {{ border-left-color: #dc3545; }}
                .priority-medium {{ border-left-color: #ffc107; }}
                .priority-low {{ border-left-color: #28a745; }}
            </style>
        </head>
        <body>
            <h2>Quality Assessment Report</h2>
            
            <div class="metric-group">
                <h3>Overall Assessment</h3>
                <table>
                    <tr>
                        <td><b>Overall Quality Score:</b></td>
                        <td>{quality_score:.1%}</td>
                    </tr>
                    <tr>
                        <td><b>Predicted mAP@0.5:</b></td>
                        <td>{predicted_map:.1%}</td>
                    </tr>
                    <tr>
                        <td class="good">✓ Good: {good_count}</td>
                        <td class="warning">⚠ Warning: {warning_count}</td>
                        <td class="error">✗ Error: {error_count}</td>
                    </tr>
                </table>
            </div>
            
            <div class="metric-group">
                <h3>Session Statistics</h3>
                <table>
                    <tr><td>Total Images:</td><td>{stats['total_images']}</td></tr>
                    <tr><td>Processed:</td><td>{stats['processed_images']}</td></tr>
                    <tr><td>Total Proposals:</td><td>{stats['total_proposals']}</td></tr>
                    <tr><td>Approved:</td><td>{stats['approved']}</td></tr>
                    <tr><td>Modified:</td><td>{stats['modified']}</td></tr>
                </table>
            </div>
            
            <div class="metric-group">
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
            </div>
            
            <div class="metric-group">
                <h3>Actionable Recommendations</h3>
                {self._generate_recommendations()}
            </div>
            
            <div class="metric-group">
                <h3>Training Configuration Suggestions</h3>
                {self._generate_training_suggestions()}
            </div>
            
            <div class="metric-group">
                <h3>Data Augmentation Recommendations</h3>
                {self._generate_augmentation_suggestions()}
            </div>
            
            <div class="metric-group">
                <h3>Priority Review Images</h3>
                {self._generate_priority_review_list()}
            </div>
        </body>
        </html>
        """
        
        self._report_view.setHtml(html)
        
    def _calculate_overall_quality_score(self) -> float:
        """Calculate overall quality score from metrics."""
        if not self.metrics:
            return 0.0
            
        # Weight different metrics
        weights = {
            "Coverage": 0.15,
            "Training Readiness": 0.20,
            "Class Balance": 0.15,
            "Data Diversity": 0.15,
            "Class Confidence": 0.10,
            "Modification Rate": 0.10,
            "Aspect Ratio Diversity": 0.05,
            "Annotation Density": 0.10
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for metric in self.metrics:
            weight = weights.get(metric.name, 0.05)
            # Convert to 0-1 score
            if metric.status == QualityStatus.GOOD:
                score = 1.0
            elif metric.status == QualityStatus.WARNING:
                score = 0.6
            else:
                score = 0.3
                
            weighted_sum += score * weight
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _predict_model_performance(self) -> float:
        """Predict likely mAP based on dataset quality."""
        # Get key metrics for prediction
        metrics_dict = {m.name: m for m in self.metrics}
        
        # Base prediction factors
        factors = []
        
        # Training readiness factor
        if "Training Readiness" in metrics_dict:
            factors.append(metrics_dict["Training Readiness"].value)
            
        # Class balance factor
        if "Class Balance" in metrics_dict:
            factors.append(metrics_dict["Class Balance"].value)
            
        # Confidence distribution factor
        if "High Confidence Ratio" in metrics_dict:
            factors.append(metrics_dict["High Confidence Ratio"].value)
            
        # Data diversity factor
        if "Data Diversity" in metrics_dict:
            factors.append(metrics_dict["Data Diversity"].value)
            
        if not factors:
            return 0.5
            
        # Calculate predicted mAP (rough estimation)
        avg_factor = np.mean(factors)
        
        # Map to typical mAP ranges
        if avg_factor >= 0.8:
            return 0.75 + (avg_factor - 0.8) * 0.75  # 75-90%
        elif avg_factor >= 0.6:
            return 0.60 + (avg_factor - 0.6) * 0.75  # 60-75%
        else:
            return avg_factor * 0.6  # 0-60%
    
    def _generate_recommendations(self) -> str:
        """Generate prioritized recommendations."""
        recommendations = []
        
        # Analyze metrics and generate recommendations
        for metric in self.metrics:
            if metric.status in [QualityStatus.WARNING, QualityStatus.ERROR]:
                priority = "high" if metric.status == QualityStatus.ERROR else "medium"
                
                if "Coverage" in metric.name:
                    recommendations.append((priority, "Dataset Coverage", 
                        "Process remaining images to achieve at least 95% coverage"))
                elif "Training Readiness" in metric.name:
                    recommendations.append((priority, "Insufficient Data",
                        "Add more annotated examples, especially for underrepresented classes"))
                elif "Uncertainty" in metric.name and metric.value >= 0.2:
                    recommendations.append(("low", "Active Learning Opportunity",
                        f"Review {int(metric.value * self.session.total_images)} uncertain images for manual annotation"))
                elif "Edge Cases" in metric.name:
                    recommendations.append((priority, "Edge Case Review",
                        "Manually review detected edge cases to improve model robustness"))
                elif "Class Confidence" in metric.name:
                    recommendations.append((priority, "Low Confidence Classes",
                        "Focus on collecting more diverse examples for poorly detected classes"))
                elif "Modification Rate" in metric.name:
                    recommendations.append((priority, "Model Accuracy",
                        "Consider retraining with corrected annotations or adjusting confidence thresholds"))
        
        # Sort by priority
        recommendations.sort(key=lambda x: 0 if x[0] == "high" else (1 if x[0] == "medium" else 2))
        
        # Generate HTML
        html = ""
        for priority, title, desc in recommendations:
            html += f'<div class="recommendation priority-{priority}"><b>{title}:</b> {desc}</div>'
            
        if not recommendations:
            html = '<div class="recommendation priority-low"><b>Great job!</b> Your dataset meets quality standards.</div>'
            
        return html
    
    def _identify_weak_classes(self) -> List[str]:
        """Identify classes with low confidence scores."""
        class_confidences = defaultdict(list)
        
        for proposals in self.session.proposals.values():
            for prop in proposals:
                if prop.is_approved:
                    class_confidences[prop.class_id].append(prop.confidence)
        
        weak_classes = []
        for class_id, confidences in class_confidences.items():
            avg_conf = np.mean(confidences)
            if avg_conf < 0.5:
                # Get class name if available
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                weak_classes.append((class_name, avg_conf))
        
        # Sort by confidence (lowest first)
        weak_classes.sort(key=lambda x: x[1])
        
        return [name for name, _ in weak_classes]
    
    def _generate_training_suggestions(self) -> str:
        """Generate training configuration suggestions."""
        stats = self.session.get_statistics()
        total_annotations = stats['approved']
        
        suggestions = []
        
        # Batch size suggestion
        if total_annotations < 1000:
            batch_size = 8
            suggestions.append(f"<li><b>Batch Size:</b> {batch_size} (small dataset)</li>")
        elif total_annotations < 10000:
            batch_size = 16
            suggestions.append(f"<li><b>Batch Size:</b> {batch_size} (medium dataset)</li>")
        else:
            batch_size = 32
            suggestions.append(f"<li><b>Batch Size:</b> {batch_size} (large dataset)</li>")
            
        # Epochs suggestion
        quality_score = self._calculate_overall_quality_score()
        if quality_score < 0.6:
            epochs = 300
            suggestions.append(f"<li><b>Epochs:</b> {epochs} (lower quality data needs more training)</li>")
        else:
            epochs = 100
            suggestions.append(f"<li><b>Epochs:</b> {epochs} (good quality data)</li>")
            
        # Model size suggestion
        if total_annotations < 5000:
            suggestions.append("<li><b>Model:</b> YOLOv8n or YOLOv8s (small dataset)</li>")
        else:
            suggestions.append("<li><b>Model:</b> YOLOv8m or YOLOv8l (sufficient data)</li>")
            
        # Learning rate
        suggestions.append("<li><b>Learning Rate:</b> 0.01 (default)</li>")
        
        # Early stopping
        suggestions.append("<li><b>Patience:</b> 50 epochs (early stopping)</li>")
        
        return "<ul>" + "".join(suggestions) + "</ul>"
    
    def _generate_augmentation_suggestions(self) -> str:
        """Generate data augmentation suggestions."""
        suggestions = []
        metrics_dict = {m.name: m for m in self.metrics}
        
        # Check aspect ratio diversity
        if "Aspect Ratio Diversity" in metrics_dict:
            if metrics_dict["Aspect Ratio Diversity"].value < 0.3:
                suggestions.append("<li><b>Rotation:</b> Enable rotation augmentation to increase shape variety</li>")
                suggestions.append("<li><b>Shear:</b> Add shear transformations for more diverse perspectives</li>")
                
        # Check data diversity
        if "Data Diversity" in metrics_dict:
            if metrics_dict["Data Diversity"].value < 0.6:
                suggestions.append("<li><b>Mosaic:</b> Enable mosaic augmentation for better context learning</li>")
                suggestions.append("<li><b>MixUp:</b> Use MixUp to create synthetic training examples</li>")
                
        # Check annotation density
        if "Annotation Density" in metrics_dict:
            if metrics_dict["Annotation Density"].value < 2:
                suggestions.append("<li><b>Copy-Paste:</b> Use copy-paste augmentation to increase object density</li>")
                
        # General recommendations
        suggestions.append("<li><b>HSV:</b> Adjust hue, saturation, value for color variations</li>")
        suggestions.append("<li><b>Flip:</b> Horizontal flip (and vertical if applicable)</li>")
        suggestions.append("<li><b>Scale:</b> Random scaling (0.5-1.5x) for size variations</li>")
        
        return "<ul>" + "".join(suggestions) + "</ul>"
    
    def _generate_priority_review_list(self) -> str:
        """Generate list of images that should be prioritized for manual review."""
        priority_images = []
        
        # Find uncertain images (confidence 0.3-0.7)
        uncertain_images = []
        edge_case_images = []
        low_conf_images = []
        
        for img_path, proposals in self.session.proposals.items():
            # Check for uncertainty
            has_uncertain = any(0.3 <= p.confidence <= 0.7 for p in proposals)
            if has_uncertain:
                avg_conf = np.mean([p.confidence for p in proposals])
                uncertain_images.append((img_path, avg_conf))
                
            # Check for edge cases
            if len(proposals) == 0 or len(proposals) > 20:
                edge_case_images.append(img_path)
            else:
                # Check for unusual box sizes
                for prop in proposals:
                    x, y, w, h = prop.bbox
                    if w * h < 400 or w * h > 100000:
                        edge_case_images.append(img_path)
                        break
                        
            # Check for low confidence
            if proposals and all(p.confidence < 0.5 for p in proposals):
                low_conf_images.append(img_path)
        
        # Sort uncertain images by confidence
        uncertain_images.sort(key=lambda x: x[1])
        
        html = "<div style='max-height: 200px; overflow-y: auto;'>"
        
        if uncertain_images:
            html += "<p><b>Uncertain Predictions (best for active learning):</b></p><ul>"
            for img_path, conf in uncertain_images[:10]:  # Show top 10
                img_name = Path(img_path).name
                html += f"<li>{img_name} (avg conf: {conf:.2f})</li>"
            if len(uncertain_images) > 10:
                html += f"<li><i>...and {len(uncertain_images)-10} more</i></li>"
            html += "</ul>"
            
        if edge_case_images:
            html += "<p><b>Edge Cases:</b></p><ul>"
            for img_path in edge_case_images[:5]:
                img_name = Path(img_path).name
                html += f"<li>{img_name}</li>"
            if len(edge_case_images) > 5:
                html += f"<li><i>...and {len(edge_case_images)-5} more</i></li>"
            html += "</ul>"
            
        if not uncertain_images and not edge_case_images:
            html += "<p>No priority review images identified. Model performance appears consistent.</p>"
            
        html += "</div>"
        
        return html
    
    @pyqtSlot()
    def _export_report(self):
        """Export report to file."""
        options = "HTML Files (*.html);;JSON Files (*.json);;Text Files (*.txt)"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Report", "quality_report.html", options
        )
        
        if file_path:
            if selected_filter == "JSON Files (*.json)":
                self._export_json(file_path)
            elif selected_filter == "Text Files (*.txt)":
                self._export_text(file_path)
            else:
                with open(file_path, 'w') as f:
                    f.write(self._report_view.toHtml())
                    
    def _export_json(self, file_path: str):
        """Export report as JSON."""
        data = {
            "session": {
                "folder_path": self.session.folder_path,
                "total_images": self.session.total_images,
                "processed_images": self.session.processed_images,
                "statistics": self.session.get_statistics()
            },
            "metrics": [
                {
                    "name": m.name,
                    "value": float(m.value),
                    "status": m.status.value,
                    "description": m.description
                }
                for m in self.metrics
            ],
            "quality_score": self._calculate_overall_quality_score(),
            "predicted_map": self._predict_model_performance()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _export_text(self, file_path: str):
        """Export report as plain text."""
        lines = [
            "QUALITY ASSESSMENT REPORT",
            "=" * 50,
            "",
            f"Overall Quality Score: {self._calculate_overall_quality_score():.1%}",
            f"Predicted mAP@0.5: {self._predict_model_performance():.1%}",
            "",
            "METRICS:",
            "-" * 50
        ]
        
        for metric in self.metrics:
            status_char = "✓" if metric.status == QualityStatus.GOOD else ("⚠" if metric.status == QualityStatus.WARNING else "✗")
            lines.append(f"{status_char} {metric.name}: {metric.value:.2f} - {metric.description}")
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))