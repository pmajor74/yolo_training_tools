"""Image processing and annotation handling for auto-annotation mode."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PyQt6.QtWidgets import QMessageBox, QApplication, QFileDialog
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QRectF
from PyQt6.QtGui import QPixmap, QCursor
from PyQt6.QtCore import Qt

from ..widgets.annotation_canvas import Annotation
from ..utils.auto_annotation_manager import AutoAnnotationManager, AnnotationProposal
from .auto_annotation_data_classes import ConfidenceCategory, SessionStats


class ImageProcessor(QObject):
    """Handles image processing and annotation operations for auto-annotation mode."""
    
    # Signals
    processingStarted = pyqtSignal(str)  # folder path
    processingProgress = pyqtSignal(int, int)  # current, total
    processingCompleted = pyqtSignal()
    imageLoaded = pyqtSignal(str)  # image path
    statsUpdated = pyqtSignal(SessionStats)
    proposalsUpdated = pyqtSignal(str)  # image path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self._annotation_manager: Optional[AutoAnnotationManager] = None
        self._current_folder: Optional[Path] = None
        self._session_stats = SessionStats()
        self._is_processing = False
        self._all_processed_images: List[str] = []
        self._annotation_proposals: Dict[str, List[AnnotationProposal]] = {}
        self._modified_annotations: Dict[str, List[Annotation]] = {}
        self._image_dimensions_cache: Dict[str, Tuple[int, int]] = {}
        self._annotation_categories: Dict[ConfidenceCategory, Set[str]] = {
            ConfidenceCategory.AUTO_APPROVED: set(),
            ConfidenceCategory.REQUIRES_REVIEW: set(),
            ConfidenceCategory.REJECTED: set(),
            ConfidenceCategory.NO_DETECTIONS: set()
        }
        
    @property
    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self._is_processing
        
    @property
    def annotation_manager(self) -> Optional[AutoAnnotationManager]:
        """Get annotation manager."""
        return self._annotation_manager
        
    @property
    def session_stats(self) -> SessionStats:
        """Get current session statistics."""
        return self._session_stats
        
    @property
    def all_processed_images(self) -> List[str]:
        """Get all processed image paths."""
        return self._all_processed_images
        
    @property
    def modified_annotations(self) -> Dict[str, List[Annotation]]:
        """Get modified annotations."""
        return self._modified_annotations
        
    @property
    def annotation_categories(self) -> Dict[ConfidenceCategory, Set[str]]:
        """Get annotation categories."""
        return self._annotation_categories
    
    def start_processing(self, folder: Path, include_annotated: bool,
                        high_threshold: float, med_threshold: float,
                        augment: bool = False, augment_settings: Optional[Dict] = None):
        """
        Start auto-annotation processing.
        
        Args:
            folder: Folder containing images
            include_annotated: Whether to include already annotated images
            high_threshold: High confidence threshold
            med_threshold: Medium confidence threshold
            augment: Whether to use augmentation
            augment_settings: Augmentation settings if enabled
        """
        self._current_folder = folder
        
        # Reset state for new session
        self._all_processed_images.clear()
        self._modified_annotations.clear()
        for category_set in self._annotation_categories.values():
            category_set.clear()
        
        # Set busy cursor
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        
        try:
            # Create annotation manager
            self._annotation_manager = AutoAnnotationManager()
            
            # Create session
            session = self._annotation_manager.create_session(
                str(self._current_folder),
                high_threshold,
                med_threshold
            )
            
            # Get image paths
            image_paths = self._get_image_paths(folder, include_annotated)
            
            if not image_paths:
                QApplication.restoreOverrideCursor()
                return
            
            # Update stats with actual count of images to process
            self._session_stats = SessionStats(
                total_images=len(image_paths),
                processed_images=0,
                auto_approved=0,
                requires_review=0,
                rejected=0,
                no_detections=0,
                modified=0,
                exported=0
            )
            self.statsUpdated.emit(self._session_stats)
            
            # Store all processed images
            self._all_processed_images = [str(p) for p in image_paths]
            
            # Pre-cache image dimensions for better performance later
            self._cache_image_dimensions()
            
            # Connect signals
            self._annotation_manager.inferenceProgress.connect(self._on_inference_progress)
            self._annotation_manager.inferenceCompleted.connect(self._on_inference_completed)
            self._annotation_manager.proposalsUpdated.connect(self._on_proposals_updated)
            
            # Start inference
            self._is_processing = True
            self.processingStarted.emit(str(folder))
            
            self._annotation_manager.start_inference(
                self._all_processed_images,
                augment=augment,
                augment_settings=augment_settings
            )
            
        except Exception as e:
            # Restore cursor on error
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self._parent, "Error", f"Failed to start auto-annotation: {str(e)}")
            self._is_processing = False
            
    def stop_processing(self):
        """Stop auto-annotation processing."""
        if self._annotation_manager:
            self._annotation_manager.stop_inference()
        self._is_processing = False
        
        # Restore cursor
        QApplication.restoreOverrideCursor()
        
    def _get_image_paths(self, folder: Path, include_annotated: bool) -> List[Path]:
        """Get image paths from folder, optionally filtering annotated ones."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        all_image_paths_set = set()  # Use set to avoid duplicates
        for ext in image_extensions:
            all_image_paths_set.update(folder.glob(f'*{ext}'))
            all_image_paths_set.update(folder.glob(f'*{ext.upper()}'))
        all_image_paths = list(all_image_paths_set)
        
        # Filter out images that already have annotations if requested
        if not include_annotated:
            image_paths = []
            skipped_count = 0
            for img_path in all_image_paths:
                # Check if corresponding .txt file exists
                annotation_file = img_path.with_suffix('.txt')
                if not annotation_file.exists():
                    image_paths.append(img_path)
                else:
                    skipped_count += 1
            
            # Don't show popup, just log the information
            if skipped_count > 0:
                print(f"Skipped {skipped_count} image(s) that already have annotations. "
                      f"Processing {len(image_paths)} unannotated image(s).")
            
            # Check if there are any images to process
            if not image_paths:
                QMessageBox.warning(self._parent, "No Images", 
                                  "All images in this folder already have annotations.\n"
                                  "Please select a different folder or check 'Include already annotated images'.")
                return []
        else:
            # Include all images
            image_paths = all_image_paths
            
        return image_paths
        
    def _cache_image_dimensions(self):
        """Pre-cache image dimensions for better performance."""
        for i, img_path in enumerate(self._all_processed_images):
            if img_path not in self._image_dimensions_cache:
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    self._image_dimensions_cache[img_path] = (pixmap.width(), pixmap.height())
            # Process events periodically to keep UI responsive
            if i % 10 == 0:
                QApplication.processEvents()
                
    def get_image_dimensions(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Get cached image dimensions."""
        return self._image_dimensions_cache.get(image_path)
        
    def load_image_for_editing(self, image_path: str) -> QPixmap:
        """
        Load image for editing.
        
        Args:
            image_path: Path to image
            
        Returns:
            QPixmap of the image
        """
        self.imageLoaded.emit(image_path)
        return QPixmap(image_path)
        
    def get_annotations_for_image(self, image_path: str) -> List[Annotation]:
        """
        Get annotations for an image (modified or proposals).
        
        Args:
            image_path: Path to image
            
        Returns:
            List of annotations
        """
        # Check if we have modified annotations for this image
        if image_path in self._modified_annotations:
            return self._modified_annotations[image_path]
        
        # Otherwise, load proposals if available
        annotations = []
        if self._annotation_manager and self._annotation_manager.current_session:
            proposals = self._annotation_manager.current_session.proposals.get(image_path, [])
            
            # Convert proposals to canvas annotations
            for i, prop in enumerate(proposals):
                ann = Annotation(
                    class_id=prop.class_id,
                    rect=QRectF(prop.bbox[0], prop.bbox[1], prop.bbox[2], prop.bbox[3]),
                    confidence=prop.confidence
                )
                annotations.append(ann)
                
        return annotations
        
    def save_annotations_for_image(self, image_path: str, annotations: List[Annotation]):
        """
        Save annotations for an image.
        
        Args:
            image_path: Path to image
            annotations: List of annotations to save
        """
        # Check if this is a new modification
        if image_path not in self._modified_annotations:
            self._session_stats.modified += 1
            self.statsUpdated.emit(self._session_stats)
            
        # Store modified annotations
        self._modified_annotations[image_path] = annotations
        
        # Update proposals in the annotation manager
        if self._annotation_manager and self._annotation_manager.current_session:
            # Convert canvas annotations back to proposals
            new_proposals = []
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                for ann in annotations:
                    rect = ann.rect
                    proposal = AnnotationProposal(
                        class_id=ann.class_id,
                        bbox=(rect.x(), rect.y(), rect.width(), rect.height()),
                        confidence=ann.confidence if ann.confidence else 1.0,
                        image_path=image_path
                    )
                    
                    # Preserve approval status if this was an existing proposal
                    category = self.get_image_category(image_path)
                    if category == ConfidenceCategory.AUTO_APPROVED:
                        proposal.is_approved = True
                    
                    new_proposals.append(proposal)
            
            # Update the proposals
            self._annotation_manager.current_session.proposals[image_path] = new_proposals
            
    def approve_proposals(self, image_paths: List[str]):
        """Approve proposals for given images."""
        for image_path in image_paths:
            if self._annotation_manager:
                self._annotation_manager.approve_proposals(image_path)
                
                # Update category
                self._update_image_category(image_path, ConfidenceCategory.AUTO_APPROVED)
                
    def reject_proposals(self, image_paths: List[str]):
        """Reject proposals for given images."""
        for image_path in image_paths:
            if self._annotation_manager:
                self._annotation_manager.reject_proposals(image_path)
                
                # Update category
                self._update_image_category(image_path, ConfidenceCategory.REJECTED)
                
    def _update_image_category(self, image_path: str, new_category: ConfidenceCategory):
        """Update the category for an image."""
        # Remove from all categories
        for cat_set in self._annotation_categories.values():
            cat_set.discard(image_path)
            
        # Add to new category
        self._annotation_categories[new_category].add(image_path)
        
    def get_image_category(self, image_path: str) -> Optional[ConfidenceCategory]:
        """Get the category for an image."""
        for category, paths in self._annotation_categories.items():
            if image_path in paths:
                return category
        return None
        
    def categorize_images(self):
        """Categorize all images based on confidence thresholds."""
        if not self._annotation_manager or not self._annotation_manager.current_session:
            return
            
        session = self._annotation_manager.current_session
        
        # Clear existing categories
        for category_set in self._annotation_categories.values():
            category_set.clear()
            
        # Categorize each image
        for image_path in self._all_processed_images:
            proposals = session.proposals.get(image_path, [])
            
            if not proposals:
                self._annotation_categories[ConfidenceCategory.NO_DETECTIONS].add(image_path)
            else:
                # Check if all proposals meet high threshold
                all_high = all(p.confidence >= session.high_threshold for p in proposals)
                # Check if any proposal is below medium threshold
                any_low = any(p.confidence < session.medium_threshold for p in proposals)
                
                if all_high:
                    self._annotation_categories[ConfidenceCategory.AUTO_APPROVED].add(image_path)
                elif any_low:
                    self._annotation_categories[ConfidenceCategory.REJECTED].add(image_path)
                else:
                    self._annotation_categories[ConfidenceCategory.REQUIRES_REVIEW].add(image_path)
                    
        # Recalculate stats
        self._recalculate_stats()
        self.statsUpdated.emit(self._session_stats)
        
    def _recalculate_stats(self):
        """Recalculate session statistics based on current categories."""
        self._session_stats.auto_approved = len(self._annotation_categories[ConfidenceCategory.AUTO_APPROVED])
        self._session_stats.requires_review = len(self._annotation_categories[ConfidenceCategory.REQUIRES_REVIEW])
        self._session_stats.rejected = len(self._annotation_categories[ConfidenceCategory.REJECTED])
        self._session_stats.no_detections = len(self._annotation_categories[ConfidenceCategory.NO_DETECTIONS])
        
    def export_annotations(self, output_folder: str, selected_paths: List[str]) -> int:
        """
        Export annotations for selected images.
        
        Args:
            output_folder: Folder to export to
            selected_paths: List of image paths to export
            
        Returns:
            Number of exported annotations
        """
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        exported_count = 0
        
        for image_path in selected_paths:
            annotations_to_export = []
            
            # First priority: Check for modified annotations (user has edited them)
            if image_path in self._modified_annotations:
                annotations_to_export = self._modified_annotations[image_path]
            # Second priority: Check for proposals from auto-annotation
            elif self._annotation_manager and self._annotation_manager.current_session:
                proposals = self._annotation_manager.current_session.proposals.get(image_path, [])
                # Convert proposals to annotations
                for prop in proposals:
                    ann = Annotation(
                        class_id=prop.class_id,
                        rect=QRectF(prop.bbox[0], prop.bbox[1], prop.bbox[2], prop.bbox[3]),
                        confidence=prop.confidence
                    )
                    annotations_to_export.append(ann)
            
            if annotations_to_export:
                # Export annotations
                self._export_single_annotation(output_path, image_path, annotations_to_export)
                exported_count += 1
                
        return exported_count
        
    def _export_single_annotation(self, output_path: Path, image_path: str, 
                                 annotations: List[Annotation]):
        """Export annotations for a single image."""
        # Get image dimensions
        dimensions = self.get_image_dimensions(image_path)
        if not dimensions:
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                return
            dimensions = (pixmap.width(), pixmap.height())
            
        img_width, img_height = dimensions
        
        # Create annotation file
        img_name = Path(image_path).stem
        ann_file = output_path / f"{img_name}.txt"
        
        with open(ann_file, 'w') as f:
            for ann in annotations:
                # Convert to YOLO format (normalized)
                rect = ann.rect
                x_center = (rect.x() + rect.width() / 2) / img_width
                y_center = (rect.y() + rect.height() / 2) / img_height
                w_norm = rect.width() / img_width
                h_norm = rect.height() / img_height
                
                # Write in YOLO format
                f.write(f"{ann.class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                
    @pyqtSlot(int, int)
    def _on_inference_progress(self, current: int, total: int):
        """Handle inference progress."""
        self._session_stats.processed_images = current
        self.processingProgress.emit(current, total)
        self.statsUpdated.emit(self._session_stats)
        
    @pyqtSlot()
    def _on_inference_completed(self):
        """Handle inference completion."""
        self._is_processing = False
        self.processingCompleted.emit()
        
        # Restore cursor
        QApplication.restoreOverrideCursor()
        
        # Categorize images based on results
        self.categorize_images()
        
    @pyqtSlot(str)
    def _on_proposals_updated(self, image_path: str):
        """Handle proposals updated for an image."""
        self.proposalsUpdated.emit(image_path)
        
    def clean_up_after_export(self, exported_paths: List[str]):
        """Clean up internal tracking after successful export."""
        for image_path in exported_paths:
            # Remove from all internal tracking structures
            if image_path in self._all_processed_images:
                self._all_processed_images.remove(image_path)
            
            # Remove from modified annotations
            self._modified_annotations.pop(image_path, None)
            
            # Remove from all confidence categories
            for category_set in self._annotation_categories.values():
                category_set.discard(image_path)
            
            # Remove from annotation proposals
            if self._annotation_manager and self._annotation_manager.current_session:
                self._annotation_manager.current_session.proposals.pop(image_path, None)
            
            # Remove from dimensions cache
            self._image_dimensions_cache.pop(image_path, None)
        
        # Update stats
        self._recalculate_stats()
        if self._session_stats.total_images >= len(exported_paths):
            self._session_stats.total_images -= len(exported_paths)
        self.statsUpdated.emit(self._session_stats)