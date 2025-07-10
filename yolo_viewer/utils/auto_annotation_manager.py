"""Auto-annotation manager for batch inference and proposal management."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from PyQt6.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
import numpy as np

from ..core import ModelCache


class ConfidenceLevel(Enum):
    """Confidence levels for annotation proposals."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AnnotationProposal:
    """Represents a proposed annotation from auto-annotation."""
    class_id: int
    bbox: Tuple[float, float, float, float]  # x, y, w, h in pixels
    confidence: float
    image_path: str
    confidence_level: ConfidenceLevel = field(init=False)
    is_approved: bool = False
    is_modified: bool = False
    
    def __post_init__(self):
        """Determine confidence level after initialization."""
        if self.confidence >= 0.8:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence >= 0.4:
            self.confidence_level = ConfidenceLevel.MEDIUM
        else:
            self.confidence_level = ConfidenceLevel.LOW
    
    def to_yolo_format(self, img_width: int, img_height: int) -> str:
        """Convert to YOLO annotation format."""
        x, y, w, h = self.bbox
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


@dataclass
class AutoAnnotationSession:
    """Represents an auto-annotation session."""
    session_id: str
    folder_path: str
    created_at: datetime
    total_images: int
    processed_images: int = 0
    proposals: Dict[str, List[AnnotationProposal]] = field(default_factory=dict)
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.4
    
    @property
    def high_threshold(self) -> float:
        """Alias for high_confidence_threshold."""
        return self.high_confidence_threshold
    
    @high_threshold.setter
    def high_threshold(self, value: float):
        """Setter for high_confidence_threshold."""
        self.high_confidence_threshold = value
    
    @property
    def medium_threshold(self) -> float:
        """Alias for medium_confidence_threshold."""
        return self.medium_confidence_threshold
    
    @medium_threshold.setter
    def medium_threshold(self, value: float):
        """Setter for medium_confidence_threshold."""
        self.medium_confidence_threshold = value
    
    def get_statistics(self) -> Dict[str, int]:
        """Get session statistics."""
        stats = {
            'total_images': self.total_images,
            'processed_images': self.processed_images,
            'total_proposals': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'approved': 0,
            'modified': 0
        }
        
        for proposals in self.proposals.values():
            for proposal in proposals:
                stats['total_proposals'] += 1
                
                if proposal.confidence_level == ConfidenceLevel.HIGH:
                    stats['high_confidence'] += 1
                elif proposal.confidence_level == ConfidenceLevel.MEDIUM:
                    stats['medium_confidence'] += 1
                else:
                    stats['low_confidence'] += 1
                
                if proposal.is_approved:
                    stats['approved'] += 1
                if proposal.is_modified:
                    stats['modified'] += 1
        
        return stats
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'folder_path': self.folder_path,
            'created_at': self.created_at.isoformat(),
            'total_images': self.total_images,
            'processed_images': self.processed_images,
            'high_confidence_threshold': self.high_confidence_threshold,
            'medium_confidence_threshold': self.medium_confidence_threshold,
            'proposals': {
                img_path: [
                    {
                        'class_id': p.class_id,
                        'bbox': p.bbox,
                        'confidence': p.confidence,
                        'is_approved': p.is_approved,
                        'is_modified': p.is_modified
                    }
                    for p in proposals
                ]
                for img_path, proposals in self.proposals.items()
            }
        }


class InferenceWorker(QThread):
    """Worker thread for batch inference."""
    
    # Signals
    progress = pyqtSignal(int, int)  # current, total
    imageProcessed = pyqtSignal(str, list)  # image_path, proposals
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, image_paths: List[str], 
                 high_threshold: float, 
                 medium_threshold: float,
                 augment: bool = False,
                 augment_settings: Optional[Dict[str, float]] = None):
        super().__init__()
        self.image_paths = image_paths
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.augment = augment
        self.augment_settings = augment_settings or {}
        self._is_running = True
        
    def run(self):
        """Run batch inference."""
        model_cache = ModelCache()
        model = model_cache.get_model()
        
        if not model:
            self.error.emit("No model loaded")
            return
        
        try:
            for i, image_path in enumerate(self.image_paths):
                if not self._is_running:
                    break
                    
                # Run inference with optional augmentation
                if self.augment:
                    results = model(image_path, augment=True, **self.augment_settings)
                else:
                    results = model(image_path)
                
                # Extract proposals
                proposals = []
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    for j in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                        confidence = float(boxes.conf[j])
                        class_id = int(boxes.cls[j])
                        
                        proposal = AnnotationProposal(
                            class_id=class_id,
                            bbox=(x1, y1, x2 - x1, y2 - y1),
                            confidence=confidence,
                            image_path=image_path
                        )
                        
                        # Auto-approve high confidence
                        if confidence >= self.high_threshold:
                            proposal.is_approved = True
                            
                        proposals.append(proposal)
                
                self.imageProcessed.emit(image_path, proposals)
                self.progress.emit(i + 1, len(self.image_paths))
                
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
    
    def stop(self):
        """Stop the worker."""
        self._is_running = False


class AutoAnnotationManager(QObject):
    """Manages auto-annotation workflow."""
    
    # Signals
    sessionCreated = pyqtSignal(str)  # session_id
    inferenceProgress = pyqtSignal(int, int)  # current, total
    inferenceCompleted = pyqtSignal()
    proposalsUpdated = pyqtSignal(str)  # image_path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_session: Optional[AutoAnnotationSession] = None
        self._worker: Optional[InferenceWorker] = None
        
    def create_session(self, folder_path: str, 
                      high_threshold: float = 0.8,
                      medium_threshold: float = 0.4) -> AutoAnnotationSession:
        """Create a new auto-annotation session."""
        # Find images in folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        
        folder = Path(folder_path)
        for ext in image_extensions:
            image_paths.extend(folder.glob(f'*{ext}'))
            image_paths.extend(folder.glob(f'*{ext.upper()}'))
        
        # Create session
        session = AutoAnnotationSession(
            session_id=datetime.now().strftime('%Y%m%d_%H%M%S'),
            folder_path=folder_path,
            created_at=datetime.now(),
            total_images=len(image_paths),
            high_confidence_threshold=high_threshold,
            medium_confidence_threshold=medium_threshold
        )
        
        self._current_session = session
        self.sessionCreated.emit(session.session_id)
        
        return session
    
    def start_inference(self, image_paths: List[str], augment: bool = False, augment_settings: Optional[Dict[str, float]] = None):
        """Start batch inference on images."""
        if not self._current_session:
            raise ValueError("No active session")
            
        self._worker = InferenceWorker(
            image_paths,
            self._current_session.high_confidence_threshold,
            self._current_session.medium_confidence_threshold,
            augment,
            augment_settings
        )
        
        self._worker.progress.connect(self._on_inference_progress)
        self._worker.imageProcessed.connect(self._on_image_processed)
        self._worker.finished.connect(self._on_inference_finished)
        self._worker.error.connect(self._on_inference_error)
        
        self._worker.start()
    
    def stop_inference(self):
        """Stop ongoing inference."""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
    
    @pyqtSlot(int, int)
    def _on_inference_progress(self, current: int, total: int):
        """Handle inference progress."""
        if self._current_session:
            self._current_session.processed_images = current
        self.inferenceProgress.emit(current, total)
    
    @pyqtSlot(str, list)
    def _on_image_processed(self, image_path: str, proposals: List[AnnotationProposal]):
        """Handle processed image."""
        if self._current_session:
            self._current_session.proposals[image_path] = proposals
            self.proposalsUpdated.emit(image_path)
    
    @pyqtSlot()
    def _on_inference_finished(self):
        """Handle inference completion."""
        self.inferenceCompleted.emit()
        self._worker = None
    
    @pyqtSlot(str)
    def _on_inference_error(self, error: str):
        """Handle inference error."""
        print(f"Inference error: {error}")
    
    def get_proposals_by_category(self, category: ConfidenceLevel) -> Dict[str, List[AnnotationProposal]]:
        """Get proposals filtered by confidence category."""
        if not self._current_session:
            return {}
            
        filtered = {}
        for image_path, proposals in self._current_session.proposals.items():
            category_proposals = [p for p in proposals if p.confidence_level == category]
            if category_proposals:
                filtered[image_path] = category_proposals
                
        return filtered
    
    def approve_proposals(self, image_path: str, proposal_indices: Optional[List[int]] = None):
        """Approve proposals for an image."""
        if not self._current_session or image_path not in self._current_session.proposals:
            return
            
        proposals = self._current_session.proposals[image_path]
        
        if proposal_indices is None:
            # Approve all
            for proposal in proposals:
                proposal.is_approved = True
        else:
            # Approve specific indices
            for idx in proposal_indices:
                if 0 <= idx < len(proposals):
                    proposals[idx].is_approved = True
                    
        self.proposalsUpdated.emit(image_path)
    
    def reject_proposals(self, image_path: str, proposal_indices: Optional[List[int]] = None):
        """Reject proposals for an image."""
        if not self._current_session or image_path not in self._current_session.proposals:
            return
            
        proposals = self._current_session.proposals[image_path]
        
        if proposal_indices is None:
            # Reject all - mark as not approved
            for proposal in proposals:
                proposal.is_approved = False
        else:
            # Reject specific indices
            for idx in proposal_indices:
                if 0 <= idx < len(proposals):
                    proposals[idx].is_approved = False
            
        self.proposalsUpdated.emit(image_path)
    
    def export_annotations(self, output_folder: str) -> int:
        """Export approved annotations to YOLO format."""
        if not self._current_session:
            return 0
            
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        exported_count = 0
        
        for image_path, proposals in self._current_session.proposals.items():
            approved = [p for p in proposals if p.is_approved]
            if not approved:
                continue
                
            # Get image dimensions (would need to load image in real implementation)
            # For now, assume standard size
            img_width, img_height = 1920, 1080  # TODO: Load actual image dimensions
            
            # Create annotation file
            image_name = Path(image_path).stem
            ann_path = output_path / f"{image_name}.txt"
            
            with open(ann_path, 'w') as f:
                for proposal in approved:
                    line = proposal.to_yolo_format(img_width, img_height)
                    f.write(line + '\n')
                    
            exported_count += 1
            
        return exported_count
    
    def save_session(self, file_path: str):
        """Save session to file."""
        if not self._current_session:
            return
            
        with open(file_path, 'w') as f:
            json.dump(self._current_session.to_dict(), f, indent=2)
    
    def load_session(self, file_path: str) -> AutoAnnotationSession:
        """Load session from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Reconstruct session
        session = AutoAnnotationSession(
            session_id=data['session_id'],
            folder_path=data['folder_path'],
            created_at=datetime.fromisoformat(data['created_at']),
            total_images=data['total_images'],
            processed_images=data['processed_images'],
            high_confidence_threshold=data['high_confidence_threshold'],
            medium_confidence_threshold=data['medium_confidence_threshold']
        )
        
        # Reconstruct proposals
        for img_path, props_data in data['proposals'].items():
            proposals = []
            for p_data in props_data:
                proposal = AnnotationProposal(
                    class_id=p_data['class_id'],
                    bbox=tuple(p_data['bbox']),
                    confidence=p_data['confidence'],
                    image_path=img_path
                )
                proposal.is_approved = p_data['is_approved']
                proposal.is_modified = p_data['is_modified']
                proposals.append(proposal)
            session.proposals[img_path] = proposals
            
        self._current_session = session
        return session
    
    @property
    def current_session(self) -> Optional[AutoAnnotationSession]:
        """Get current session."""
        return self._current_session