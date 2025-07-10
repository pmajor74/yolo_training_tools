"""Data classes and enums for auto-annotation mode."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class ConfidenceCategory(Enum):
    """Confidence categories for annotation proposals."""
    AUTO_APPROVED = "auto_approved"
    REQUIRES_REVIEW = "requires_review" 
    REJECTED = "rejected"
    NO_DETECTIONS = "no_detections"


@dataclass
class SessionStats:
    """Statistics for the current auto-annotation session."""
    total_images: int = 0
    processed_images: int = 0
    auto_approved: int = 0
    requires_review: int = 0
    rejected: int = 0
    no_detections: int = 0
    modified: int = 0
    exported: int = 0


@dataclass
class WorkflowState:
    """State tracking for the integrated workflow."""
    iteration: int = 0
    current_stage: str = "idle"  # idle, annotating, exporting, splitting, training, loading_model
    dataset_path: Optional[Path] = None
    model_history: List[Tuple[int, str]] = None  # [(iteration, model_path), ...]
    last_export_path: Optional[Path] = None
    training_in_progress: bool = False
    
    def __post_init__(self):
        if self.model_history is None:
            self.model_history = []