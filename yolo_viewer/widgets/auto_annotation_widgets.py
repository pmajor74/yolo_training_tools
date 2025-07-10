"""Custom widgets for auto-annotation mode."""

from typing import List, Optional, Dict
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFrame, QProgressDialog,
    QCheckBox, QGroupBox, QTextEdit, QDialog, QDialogButtonBox,
    QScrollArea, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QFont

from ..utils.auto_annotation_manager import AnnotationProposal, ConfidenceLevel


class CategoryStatsWidget(QWidget):
    """Widget showing statistics for each confidence category."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI."""
        layout = QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Headers
        layout.addWidget(QLabel("<b>Category</b>"), 0, 0)
        layout.addWidget(QLabel("<b>Count</b>"), 0, 1)
        layout.addWidget(QLabel("<b>Actions</b>"), 0, 2)
        
        # Auto-approved row
        self._approved_label = QLabel("Auto-Approved:")
        self._approved_label.setStyleSheet("color: #28a745;")
        layout.addWidget(self._approved_label, 1, 0)
        
        self._approved_count = QLabel("0")
        layout.addWidget(self._approved_count, 1, 1)
        
        self._approved_action = QPushButton("View")
        self._approved_action.setMaximumWidth(60)
        layout.addWidget(self._approved_action, 1, 2)
        
        # Requires review row
        self._review_label = QLabel("Requires Review:")
        self._review_label.setStyleSheet("color: #ffc107;")
        layout.addWidget(self._review_label, 2, 0)
        
        self._review_count = QLabel("0")
        layout.addWidget(self._review_count, 2, 1)
        
        self._review_action = QPushButton("View")
        self._review_action.setMaximumWidth(60)
        layout.addWidget(self._review_action, 2, 2)
        
        # Rejected row
        self._rejected_label = QLabel("Rejected:")
        self._rejected_label.setStyleSheet("color: #dc3545;")
        layout.addWidget(self._rejected_label, 3, 0)
        
        self._rejected_count = QLabel("0")
        layout.addWidget(self._rejected_count, 3, 1)
        
        self._rejected_action = QPushButton("View")
        self._rejected_action.setMaximumWidth(60)
        layout.addWidget(self._rejected_action, 3, 2)
        
    def update_stats(self, approved: int, review: int, rejected: int):
        """Update category statistics."""
        self._approved_count.setText(str(approved))
        self._review_count.setText(str(review))
        self._rejected_count.setText(str(rejected))


class ProposalListWidget(QListWidget):
    """List widget for showing annotation proposals with visual indicators."""
    
    proposalSelected = pyqtSignal(AnnotationProposal)
    proposalApproved = pyqtSignal(int)  # index
    proposalRejected = pyqtSignal(int)  # index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.itemClicked.connect(self._on_item_clicked)
        
    def add_proposal(self, proposal: AnnotationProposal, index: int):
        """Add a proposal to the list."""
        # Create item text
        text = f"Class {proposal.class_id} - {proposal.confidence:.2f}"
        if proposal.is_modified:
            text += " (Modified)"
            
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, (proposal, index))
        
        # Set color based on confidence level
        if proposal.confidence_level == ConfidenceLevel.HIGH:
            item.setForeground(QColor("#28a745"))
        elif proposal.confidence_level == ConfidenceLevel.MEDIUM:
            item.setForeground(QColor("#ffc107"))
        else:
            item.setForeground(QColor("#dc3545"))
            
        # Add checkbox for approval
        item.setCheckState(Qt.CheckState.Checked if proposal.is_approved else Qt.CheckState.Unchecked)
        
        self.addItem(item)
        
    def clear_proposals(self):
        """Clear all proposals."""
        self.clear()
        
    @pyqtSlot(QListWidgetItem)
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        proposal, index = item.data(Qt.ItemDataRole.UserRole)
        
        # Check if checkbox state changed
        is_checked = item.checkState() == Qt.CheckState.Checked
        if is_checked != proposal.is_approved:
            if is_checked:
                self.proposalApproved.emit(index)
            else:
                self.proposalRejected.emit(index)
                
        self.proposalSelected.emit(proposal)


class BatchProgressDialog(QProgressDialog):
    """Progress dialog for batch operations."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumDuration(0)
        self.setCancelButton(None)
        self.setAutoClose(True)
        self.setAutoReset(True)
        
    def set_progress(self, current: int, total: int, message: str = ""):
        """Update progress."""
        self.setMaximum(total)
        self.setValue(current)
        if message:
            self.setLabelText(message)


class AnnotationPreviewWidget(QWidget):
    """Widget for previewing image with annotation proposals."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._proposals: List[AnnotationProposal] = []
        self._selected_index: Optional[int] = None
        self.setMinimumSize(400, 300)
        
    def set_image(self, image_path: str):
        """Set the image to display."""
        self._pixmap = QPixmap(image_path)
        self.update()
        
    def set_proposals(self, proposals: List[AnnotationProposal]):
        """Set annotation proposals."""
        self._proposals = proposals
        self.update()
        
    def set_selected_proposal(self, index: Optional[int]):
        """Set selected proposal index."""
        self._selected_index = index
        self.update()
        
    def paintEvent(self, event):
        """Paint the widget."""
        if not self._pixmap:
            return
            
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw image scaled to fit
        widget_rect = self.rect()
        pixmap_scaled = self._pixmap.scaled(
            widget_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Center the image
        x = (widget_rect.width() - pixmap_scaled.width()) // 2
        y = (widget_rect.height() - pixmap_scaled.height()) // 2
        painter.drawPixmap(x, y, pixmap_scaled)
        
        # Calculate scale factors
        scale_x = pixmap_scaled.width() / self._pixmap.width()
        scale_y = pixmap_scaled.height() / self._pixmap.height()
        
        # Draw proposals
        for i, proposal in enumerate(self._proposals):
            # Skip rejected proposals
            if not proposal.is_approved and self._selected_index != i:
                continue
                
            # Set pen color based on confidence
            if proposal.confidence_level == ConfidenceLevel.HIGH:
                color = QColor("#28a745")
            elif proposal.confidence_level == ConfidenceLevel.MEDIUM:
                color = QColor("#ffc107")
            else:
                color = QColor("#dc3545")
                
            # Highlight selected
            if i == self._selected_index:
                pen = QPen(color, 3)
            else:
                pen = QPen(color, 2)
                
            painter.setPen(pen)
            
            # Draw bbox
            bbox_x = x + proposal.bbox[0] * scale_x
            bbox_y = y + proposal.bbox[1] * scale_y
            bbox_w = proposal.bbox[2] * scale_x
            bbox_h = proposal.bbox[3] * scale_y
            
            painter.drawRect(int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h))
            
            # Draw label
            label = f"Class {proposal.class_id}: {proposal.confidence:.2f}"
            painter.fillRect(int(bbox_x), int(bbox_y - 20), 
                           len(label) * 7, 20, color)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(int(bbox_x + 2), int(bbox_y - 5), label)
            painter.setPen(pen)


class SessionSummaryDialog(QDialog):
    """Dialog showing auto-annotation session summary."""
    
    def __init__(self, stats: Dict[str, int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session Summary")
        self.setModal(True)
        self._setup_ui(stats)
        
    def _setup_ui(self, stats: Dict[str, int]):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        
        # Summary text
        summary = QTextEdit()
        summary.setReadOnly(True)
        
        text = f"""
        <h3>Auto-Annotation Session Summary</h3>
        
        <table>
        <tr><td><b>Total Images:</b></td><td>{stats['total_images']}</td></tr>
        <tr><td><b>Processed Images:</b></td><td>{stats['processed_images']}</td></tr>
        <tr><td><b>Total Proposals:</b></td><td>{stats['total_proposals']}</td></tr>
        </table>
        
        <h4>Confidence Distribution</h4>
        <table>
        <tr><td>High Confidence:</td><td>{stats['high_confidence']}</td></tr>
        <tr><td>Medium Confidence:</td><td>{stats['medium_confidence']}</td></tr>
        <tr><td>Low Confidence:</td><td>{stats['low_confidence']}</td></tr>
        </table>
        
        <h4>User Actions</h4>
        <table>
        <tr><td>Approved:</td><td>{stats['approved']}</td></tr>
        <tr><td>Modified:</td><td>{stats['modified']}</td></tr>
        </table>
        """
        
        summary.setHtml(text)
        layout.addWidget(summary)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        
        self.resize(400, 400)


class ActiveLearningSampleWidget(QWidget):
    """Widget for displaying active learning sample selection."""
    
    sampleSelected = pyqtSignal(str)  # image_path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples: List[Tuple[str, float]] = []  # (image_path, score)
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("<b>Active Learning Samples</b>")
        layout.addWidget(header)
        
        # Info
        self._info_label = QLabel("Samples with highest learning value")
        self._info_label.setStyleSheet("color: #888;")
        layout.addWidget(self._info_label)
        
        # Sample list
        self._list = QListWidget()
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)
        
    def set_samples(self, samples: List[Tuple[str, float]]):
        """Set active learning samples."""
        self._samples = samples
        self._list.clear()
        
        for image_path, score in samples:
            item = QListWidgetItem(f"{Path(image_path).name} (score: {score:.3f})")
            item.setData(Qt.ItemDataRole.UserRole, image_path)
            self._list.addItem(item)
            
    @pyqtSlot(QListWidgetItem)
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        image_path = item.data(Qt.ItemDataRole.UserRole)
        self.sampleSelected.emit(image_path)