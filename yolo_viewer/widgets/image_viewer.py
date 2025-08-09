"""Image viewer widget with zoom, pan, and annotation overlay support."""

from typing import Optional, List, Tuple, Dict
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QBrush, QWheelEvent,
    QMouseEvent, QTransform, QFont
)

from ..core.constants import (
    ANNOTATION_COLORS, ANNOTATION_LINE_WIDTH, 
    BELOW_THRESHOLD_COLOR, BELOW_THRESHOLD_LINE_WIDTH,
    COLOR_MANAGER
)


class ImageViewer(QGraphicsView):
    """Enhanced image viewer with zoom/pan and annotation overlay."""
    
    # Signals
    zoomChanged = pyqtSignal(float)  # zoom level
    mousePositionChanged = pyqtSignal(QPointF)  # mouse position in image coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup scene with explicit rect to avoid null paint device
        self._scene = QGraphicsScene()
        self._scene.setSceneRect(0, 0, 1, 1)  # Set minimal rect to avoid null scene
        self.setScene(self._scene)
        
        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)  # Start with no drag
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # State
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._annotations: List[Tuple] = []
        self._confidences: List[float] = []
        self._below_threshold_flags: List[bool] = []
        self._confidence_threshold: float = 0.25
        self._class_names: Dict[int, str] = {}  # class_id -> class_name mapping
        self._zoom_factor: float = 1.0
        self._min_zoom: float = 0.1
        self._max_zoom: float = 10.0
        self._fit_on_load: bool = True
        
        # Mouse tracking for position
        self.setMouseTracking(True)
    
    def load_image(self, pixmap: QPixmap):
        """Load an image into the viewer."""
        # Clear previous
        self._scene.clear()
        
        # Add pixmap
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        
        # Fit the entire image in view while maintaining aspect ratio
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = self.transform().m11()
        self.zoomChanged.emit(self._zoom_factor)
        
        # Draw annotations if any
        if self._annotations:
            self._draw_annotations()
    
    def set_annotations(self, annotations: List[Tuple], 
                       confidences: Optional[List[float]] = None,
                       below_threshold_flags: Optional[List[bool]] = None,
                       confidence_threshold: float = 0.25):
        """
        Set annotations to display.
        
        Args:
            annotations: List of YOLO format annotations (class_id, x_center, y_center, width, height)
            confidences: Optional list of confidence scores for each annotation
            below_threshold_flags: Optional list indicating if annotation is below threshold
            confidence_threshold: Confidence threshold for display purposes
        """
        self._annotations = annotations
        self._confidences = confidences or []
        self._below_threshold_flags = below_threshold_flags or []
        self._confidence_threshold = confidence_threshold
        
        if self._pixmap_item:
            self._draw_annotations()
    
    def set_class_names(self, class_names: Dict[int, str]):
        """Set class names mapping for display."""
        self._class_names = class_names
        # Redraw if we have annotations
        if self._pixmap_item and self._annotations:
            self._draw_annotations()
    
    def clear_annotations(self):
        """Clear all annotations."""
        self._annotations = []
        self._confidences = []
        self._below_threshold_flags = []
        if self._pixmap_item:
            # Remove annotation items
            for item in self._scene.items():
                if item != self._pixmap_item:
                    self._scene.removeItem(item)
    
    def fit_to_window(self):
        """Fit image to window."""
        if not self._pixmap_item:
            return
        
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = self.transform().m11()
        self.zoomChanged.emit(self._zoom_factor)
    
    def actual_size(self):
        """Reset to actual size (1:1)."""
        self.resetTransform()
        self._zoom_factor = 1.0
        self.zoomChanged.emit(self._zoom_factor)
    
    def zoom_in(self):
        """Zoom in by 25%."""
        self.scale_view(1.25)
    
    def zoom_out(self):
        """Zoom out by 25%."""
        self.scale_view(0.8)
    
    def scale_view(self, scale_factor: float):
        """Scale the view by a factor."""
        new_zoom = self._zoom_factor * scale_factor
        
        if self._min_zoom <= new_zoom <= self._max_zoom:
            self.scale(scale_factor, scale_factor)
            self._zoom_factor = new_zoom
            self.zoomChanged.emit(self._zoom_factor)
    
    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.resetTransform()
        self._zoom_factor = 1.0
        self.zoomChanged.emit(self._zoom_factor)
    
    def get_zoom_factor(self) -> float:
        """Get current zoom factor."""
        return self._zoom_factor
    
    def set_fit_on_load(self, fit: bool):
        """Set whether to fit image to window on load."""
        self._fit_on_load = fit
    
    def _draw_annotations(self):
        """Draw annotation overlays."""
        if not self._pixmap_item:
            return
        
        # Remove existing annotation items
        for item in self._scene.items():
            if item != self._pixmap_item:
                self._scene.removeItem(item)
        
        # Get image dimensions
        pixmap = self._pixmap_item.pixmap()
        img_width = pixmap.width()
        img_height = pixmap.height()
        
        # Draw each annotation
        for i, ann in enumerate(self._annotations):
            if len(ann) >= 5:
                class_id, x_center, y_center, width, height = ann[:5]
                
                # Get confidence and below_threshold flag if available
                confidence = self._confidences[i] if i < len(self._confidences) else None
                below_threshold = self._below_threshold_flags[i] if i < len(self._below_threshold_flags) else False
                
                # Convert normalized to pixel coordinates
                x = (x_center - width / 2) * img_width
                y = (y_center - height / 2) * img_height
                w = width * img_width
                h = height * img_height
                
                # Determine color and style based on below_threshold flag
                if below_threshold:
                    color = QColor(*BELOW_THRESHOLD_COLOR)
                    line_width = BELOW_THRESHOLD_LINE_WIDTH
                    pen_style = Qt.PenStyle.SolidLine
                else:
                    # Use COLOR_MANAGER for colorblind-friendly colors
                    color = COLOR_MANAGER.get_qcolor(int(class_id))
                    pen_style = COLOR_MANAGER.get_pen_style(int(class_id))
                    line_width = 4  # Increased from ANNOTATION_LINE_WIDTH for better visibility
                
                # Draw rectangle with appropriate style
                pen = QPen(color, line_width)
                pen.setStyle(pen_style)
                pen.setCosmetic(True)  # Keep consistent width regardless of zoom
                rect = self._scene.addRect(x, y, w, h, pen)
                
                # Draw label with class name and confidence
                if confidence is not None or self._class_names:
                    # Build label text
                    label_parts = []
                    
                    # Add class name if available
                    class_name = self._class_names.get(int(class_id), f"Class {int(class_id)}")
                    label_parts.append(class_name)
                    
                    # Add confidence if available
                    if confidence is not None:
                        label_parts.append(f"{confidence:.0%}")
                    
                    label_text = " ".join(label_parts)
                    
                    # Dynamic font size based on view scale for better readability
                    view_scale = self.transform().m11()
                    if view_scale < 0.5:
                        # Very zoomed out - use larger font
                        font_size = int(16 / view_scale)
                    elif view_scale < 1.0:
                        # Moderately zoomed out - scale font appropriately
                        font_size = int(12 / view_scale)
                    else:
                        # Zoomed in or normal - use standard sizing
                        font_size = min(12, int(12 / view_scale))
                    
                    # Clamp to reasonable range
                    font_size = max(10, min(36, font_size))
                    font = QFont("Arial", font_size, QFont.Weight.Bold)
                    
                    # Measure text to size background appropriately
                    temp_text = self._scene.addText(label_text)
                    temp_text.setFont(font)
                    text_rect = temp_text.boundingRect()
                    self._scene.removeItem(temp_text)
                    
                    # Position label inside the box if it's large enough, otherwise above
                    bg_width = text_rect.width() + 8
                    bg_height = text_rect.height() + 4
                    min_box_height = bg_height * 2
                    
                    if h > min_box_height:
                        # Box is large enough - place label inside at top
                        text_bg = self._scene.addRect(x + 2, y + 2, bg_width, bg_height, 
                                                     QPen(Qt.PenStyle.NoPen), 
                                                     QBrush(color))
                        text_pos_x = x + 5
                        text_pos_y = y + 2
                    else:
                        # Box is too small - place label above
                        text_bg = self._scene.addRect(x, y - bg_height, bg_width, bg_height, 
                                                     QPen(Qt.PenStyle.NoPen), 
                                                     QBrush(color))
                        text_pos_x = x + 3
                        text_pos_y = y - bg_height
                    
                    # Draw text with better contrast
                    text = self._scene.addText(label_text)
                    # Use COLOR_MANAGER for contrasting text color
                    text_color = QColor(*COLOR_MANAGER.get_text_color(int(class_id)))
                    text.setDefaultTextColor(text_color)
                    text.setFont(font)
                    text.setPos(text_pos_x, text_pos_y)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        # Always zoom with mouse wheel
        delta = event.angleDelta().y()
        scale_factor = 1.1 if delta > 0 else 0.9
        
        # Get the position to zoom towards
        old_pos = self.mapToScene(event.position().toPoint())
        
        # Scale
        self.scale_view(scale_factor)
        new_pos = self.mapToScene(event.position().toPoint())
        
        # Adjust position to zoom towards cursor
        delta_pos = new_pos - old_pos
        self.translate(delta_pos.x(), delta_pos.y())
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning."""
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Start panning
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Create a fake event without Ctrl to pass to parent
            fake_event = QMouseEvent(event.type(), event.position(), event.globalPosition(),
                                   Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                   Qt.KeyboardModifier.NoModifier)
            super().mousePressEvent(fake_event)
        else:
            super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Track mouse position."""
        if self._pixmap_item:
            scene_pos = self.mapToScene(event.pos())
            if self._pixmap_item.contains(scene_pos):
                self.mousePositionChanged.emit(scene_pos)
        
        super().mouseMoveEvent(event)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self.zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
        elif event.key() == Qt.Key.Key_0:
            self.fit_to_window()
        elif event.key() == Qt.Key.Key_1:
            self.actual_size()
        else:
            super().keyPressEvent(event)