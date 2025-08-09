"""Interactive annotation canvas using QGraphicsView for drawing and editing bounding boxes."""

from typing import List, Optional, Tuple, Dict, Set
from enum import Enum, auto
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsPixmapItem
)
from PyQt6.QtCore import (
    Qt, QRectF, QPointF, pyqtSignal, pyqtSlot
)
from PyQt6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QBrush, QCursor,
    QMouseEvent, QKeyEvent, QWheelEvent, QFont, QTransform,
    QUndoCommand, QUndoStack, QImage
)

from ..core.constants import ANNOTATION_COLORS, ANNOTATION_LINE_WIDTH, COLOR_MANAGER


class ToolMode(Enum):
    """Annotation tool modes."""
    SELECT = auto()
    DRAW = auto()
    PAN = auto()


class HandlePosition(Enum):
    """Resize handle positions."""
    NONE = auto()
    TOP_LEFT = auto()
    TOP = auto()
    TOP_RIGHT = auto()
    LEFT = auto()
    RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM = auto()
    BOTTOM_RIGHT = auto()


@dataclass
class Annotation:
    """Annotation data."""
    class_id: int
    rect: QRectF
    id: Optional[str] = None
    confidence: Optional[float] = None
    
    def to_yolo_format(self, img_width: float, img_height: float) -> Tuple[float, ...]:
        """Convert to YOLO format (class_id, x_center, y_center, width, height)."""
        x_center = (self.rect.x() + self.rect.width() / 2) / img_width
        y_center = (self.rect.y() + self.rect.height() / 2) / img_height
        width = self.rect.width() / img_width
        height = self.rect.height() / img_height
        return (self.class_id, x_center, y_center, width, height)


class AnnotationItem(QGraphicsRectItem):
    """Custom graphics item for annotations."""
    
    def __init__(self, annotation: Annotation, canvas=None, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.canvas = canvas  # Reference to canvas for signals
        self.setRect(annotation.rect)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        
        # Visual properties
        self._update_appearance()
        
        # Resize handles
        self._handles: Dict[HandlePosition, QRectF] = {}
        self._handle_size = 8
        self._hovered_handle = HandlePosition.NONE
        self._resize_handle = HandlePosition.NONE
        self._resize_start_rect = None
        self._resize_start_pos = None
        
        # Track if we're being modified
        self._is_modifying = False
        
    def _update_appearance(self):
        """Update visual appearance based on selection state."""
        # Use COLOR_MANAGER for colorblind-friendly colors
        color = COLOR_MANAGER.get_qcolor(self.annotation.class_id)
        pen_style = COLOR_MANAGER.get_pen_style(self.annotation.class_id)
        
        # Increase line width for better visibility on high-res images
        base_width = 5 if self.isSelected() else 4
        pen = QPen(color, base_width)
        pen.setStyle(pen_style)
        pen.setCosmetic(True)  # Keep consistent width regardless of zoom
        
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
    
    def itemChange(self, change, value):
        """Handle item changes."""
        if change == QGraphicsRectItem.GraphicsItemChange.ItemSelectedChange:
            self._update_appearance()
        elif change == QGraphicsRectItem.GraphicsItemChange.ItemPositionChange:
            # Force scene update before moving
            if self.scene():
                self.scene().update()
        elif change == QGraphicsRectItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update annotation rect to match current position
            # Don't translate the rect, just update position
            new_rect = QRectF(self.pos().x(), self.pos().y(), 
                            self.annotation.rect.width(), self.annotation.rect.height())
            self.annotation.rect = new_rect
            # Force update of the item's bounding rect
            self.prepareGeometryChange()
        
        return super().itemChange(change, value)
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint the annotation."""
        super().paint(painter, option, widget)
        
        # Draw handles if selected
        if self.isSelected():
            self._draw_handles(painter)
        
        # Draw class label
        self._draw_label(painter)
    
    def _draw_handles(self, painter: QPainter):
        """Draw resize handles."""
        painter.save()
        
        # Update handle positions
        rect = self.rect()
        hs = self._handle_size / 2
        
        self._handles = {
            HandlePosition.TOP_LEFT: QRectF(rect.left() - hs, rect.top() - hs, 
                                           self._handle_size, self._handle_size),
            HandlePosition.TOP: QRectF(rect.center().x() - hs, rect.top() - hs,
                                      self._handle_size, self._handle_size),
            HandlePosition.TOP_RIGHT: QRectF(rect.right() - hs, rect.top() - hs,
                                            self._handle_size, self._handle_size),
            HandlePosition.LEFT: QRectF(rect.left() - hs, rect.center().y() - hs,
                                       self._handle_size, self._handle_size),
            HandlePosition.RIGHT: QRectF(rect.right() - hs, rect.center().y() - hs,
                                        self._handle_size, self._handle_size),
            HandlePosition.BOTTOM_LEFT: QRectF(rect.left() - hs, rect.bottom() - hs,
                                              self._handle_size, self._handle_size),
            HandlePosition.BOTTOM: QRectF(rect.center().x() - hs, rect.bottom() - hs,
                                         self._handle_size, self._handle_size),
            HandlePosition.BOTTOM_RIGHT: QRectF(rect.right() - hs, rect.bottom() - hs,
                                               self._handle_size, self._handle_size)
        }
        
        # Draw handles
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.setBrush(QBrush(Qt.GlobalColor.black))
        
        for pos, handle_rect in self._handles.items():
            if pos == self._hovered_handle:
                painter.setBrush(QBrush(Qt.GlobalColor.yellow))
            else:
                painter.setBrush(QBrush(Qt.GlobalColor.black))
            painter.drawRect(handle_rect)
        
        painter.restore()
    
    def _draw_label(self, painter: QPainter):
        """Draw class label."""
        painter.save()
        
        # Check if we should show class names or IDs
        show_names = True
        if self.canvas and hasattr(self.canvas, '_show_class_names'):
            show_names = self.canvas._show_class_names
        
        # Prepare text based on toggle
        if show_names:
            class_name = None
            if self.canvas and hasattr(self.canvas, '_class_names'):
                class_name = self.canvas._class_names.get(self.annotation.class_id)
            
            if class_name:
                text = class_name
            else:
                text = f"C{self.annotation.class_id}"
        else:
            text = f"C{self.annotation.class_id}"
            
        if self.annotation.confidence:
            text += f" {self.annotation.confidence:.0%}"
        
        # Draw background
        rect = self.rect()
        
        # Dynamic font size based on view scale for better readability
        view = self.scene().views()[0] if self.scene() and self.scene().views() else None
        if view:
            # Get the current view transform scale
            transform = view.transform()
            scale = transform.m11()  # horizontal scale factor
            # Calculate font size that's inversely proportional to zoom
            # When zoomed out (scale < 1), use larger font
            # When zoomed in (scale > 1), use smaller font
            if scale < 0.5:
                # Very zoomed out - use much larger font
                font_size = int(18 / scale)
            elif scale < 1.0:
                # Moderately zoomed out - scale font appropriately
                font_size = int(14 / scale)
            else:
                # Zoomed in or normal - use standard sizing
                font_size = min(14, int(14 / scale))
            
            # Clamp to reasonable range
            font_size = max(10, min(48, font_size))
        else:
            font_size = 14
            
        font = QFont("Arial", font_size, QFont.Weight.Bold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(text)
        
        # Position label at top-right corner to avoid covering important corners
        # This leaves the top-left corner visible for validation
        label_width = text_rect.width() + 8
        label_height = text_rect.height() + 4
        min_box_height = label_height * 2
        
        if rect.height() > min_box_height:
            # Box is large enough - place label inside at top-right
            bg_rect = QRectF(rect.right() - label_width - 2, rect.top() + 2,
                            label_width, label_height)
        else:
            # Box is too small - place label above at right side
            bg_rect = QRectF(rect.right() - label_width, rect.top() - label_height - 2,
                            label_width, label_height)
        
        # Use COLOR_MANAGER for background color
        bg_color = COLOR_MANAGER.get_qcolor(self.annotation.class_id)
        
        # Make label semi-transparent if we're resizing or moving
        if self._is_modifying or self._resize_handle != HandlePosition.NONE:
            bg_color.setAlpha(38)  # 15% opacity (255 * 0.15 = 38)
        
        painter.fillRect(bg_rect, bg_color)
        
        # Use contrasting text color for better readability
        text_color = QColor(*COLOR_MANAGER.get_text_color(self.annotation.class_id))
        
        # Also make text semi-transparent when modifying
        if self._is_modifying or self._resize_handle != HandlePosition.NONE:
            text_color.setAlpha(38)  # 15% opacity
        
        # Draw text with contrasting color
        painter.setPen(QPen(text_color))
        painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, text)
        
        painter.restore()
    
    def get_handle_at_pos(self, pos: QPointF) -> HandlePosition:
        """Get handle at position."""
        for handle_pos, rect in self._handles.items():
            if rect.contains(pos):
                return handle_pos
        return HandlePosition.NONE
    
    def boundingRect(self):
        """Return bounding rect that includes handles and label."""
        rect = super().boundingRect()
        
        # Expand for handles
        margin = self._handle_size
        
        # Expand for label at top
        label_height = 25  # Approximate height for label
        
        return rect.adjusted(-margin, -margin - label_height, margin, margin)
    
    def hoverMoveEvent(self, event):
        """Handle hover events."""
        self._hovered_handle = self.get_handle_at_pos(event.pos())
        
        # Update cursor based on handle
        if self._hovered_handle != HandlePosition.NONE:
            if self._hovered_handle in [HandlePosition.TOP_LEFT, HandlePosition.BOTTOM_RIGHT]:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif self._hovered_handle in [HandlePosition.TOP_RIGHT, HandlePosition.BOTTOM_LEFT]:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif self._hovered_handle in [HandlePosition.TOP, HandlePosition.BOTTOM]:
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif self._hovered_handle in [HandlePosition.LEFT, HandlePosition.RIGHT]:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        self.update()
        super().hoverMoveEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press for resizing."""
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.get_handle_at_pos(event.pos())
            if handle != HandlePosition.NONE:
                self._resize_handle = handle
                self._resize_start_rect = QRectF(self.rect())
                self._resize_start_pos = event.pos()
                self._is_modifying = True
                event.accept()
                return
            else:
                # Starting to move
                self._is_modifying = True
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for resizing."""
        if self._resize_handle != HandlePosition.NONE:
            # Calculate resize
            delta = event.pos() - self._resize_start_pos
            new_rect = QRectF(self._resize_start_rect)
            
            # Apply resize based on handle
            if self._resize_handle == HandlePosition.TOP_LEFT:
                new_rect.setTopLeft(new_rect.topLeft() + delta)
            elif self._resize_handle == HandlePosition.TOP:
                new_rect.setTop(new_rect.top() + delta.y())
            elif self._resize_handle == HandlePosition.TOP_RIGHT:
                new_rect.setTopRight(new_rect.topRight() + delta)
            elif self._resize_handle == HandlePosition.LEFT:
                new_rect.setLeft(new_rect.left() + delta.x())
            elif self._resize_handle == HandlePosition.RIGHT:
                new_rect.setRight(new_rect.right() + delta.x())
            elif self._resize_handle == HandlePosition.BOTTOM_LEFT:
                new_rect.setBottomLeft(new_rect.bottomLeft() + delta)
            elif self._resize_handle == HandlePosition.BOTTOM:
                new_rect.setBottom(new_rect.bottom() + delta.y())
            elif self._resize_handle == HandlePosition.BOTTOM_RIGHT:
                new_rect.setBottomRight(new_rect.bottomRight() + delta)
            
            # Ensure minimum size
            if new_rect.width() > 10 and new_rect.height() > 10:
                self.setRect(new_rect.normalized())
                self.annotation.rect = new_rect.normalized()
            
            event.accept()
            return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release for resizing."""
        was_modifying = self._is_modifying
        
        if self._resize_handle != HandlePosition.NONE:
            self._resize_handle = HandlePosition.NONE
            self._resize_start_rect = None
            self._resize_start_pos = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        
        # If we were modifying and have a canvas reference, emit the signal
        if was_modifying and self.canvas:
            # Update annotation data
            self.annotation.rect = QRectF(self.pos().x(), self.pos().y(), 
                                         self.rect().width(), self.rect().height())
            self.canvas.annotationModified.emit(self.annotation)
        
        self._is_modifying = False


class AnnotationCanvas(QGraphicsView):
    """Interactive canvas for drawing and editing annotations."""
    
    # Signals
    annotationAdded = pyqtSignal(Annotation)
    annotationModified = pyqtSignal(Annotation)
    annotationDeleted = pyqtSignal(Annotation)
    selectionChanged = pyqtSignal(list)  # List of selected annotations
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup scene with explicit rect to avoid null paint device
        self._scene = QGraphicsScene()
        self._scene.setSceneRect(0, 0, 1, 1)  # Set minimal rect to avoid null scene
        self.setScene(self._scene)
        
        # Connect to scene selection changed signal
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
        
        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Set viewport update mode - use FullViewportUpdate for better compatibility with TIF files
        # This ensures proper rendering of overlays on all image formats
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        
        # State
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._tool_mode = ToolMode.SELECT
        self._current_class_id = 0
        self._annotation_items: Dict[str, AnnotationItem] = {}
        self._is_read_only = False
        self._class_names: Dict[int, str] = {}  # Mapping of class_id to name
        self._show_class_names = True  # Toggle for showing names vs IDs
        
        # Drawing state
        self._is_drawing = False
        self._draw_start_pos = QPointF()
        self._preview_rect: Optional[QGraphicsRectItem] = None
        
        # Undo/Redo
        self._undo_stack = QUndoStack(self)
        
        # Mouse tracking
        self.setMouseTracking(True)
        
        # Initialize
        self._update_cursor()
    
    def set_tool_mode(self, mode: ToolMode):
        """Set the current tool mode."""
        self._tool_mode = mode
        self._update_cursor()
        
        if mode == ToolMode.PAN:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
    
    def set_current_class(self, class_id: int):
        """Set the current class for drawing and update selected annotations."""
        self._current_class_id = class_id
        
        # Update class for all selected annotations
        for item in self._annotation_items.values():
            if item.isSelected():
                item.annotation.class_id = class_id
                item._update_appearance()
                item.update()  # Force redraw
                self.annotationModified.emit(item.annotation)
    
    def set_read_only(self, read_only: bool):
        """Set read-only mode."""
        self._is_read_only = read_only
        if read_only:
            self.set_tool_mode(ToolMode.PAN)
    
    def set_class_names(self, class_names: Dict[int, str]):
        """Set class names for display."""
        self._class_names = class_names.copy()
        
        # Update all existing annotation items to show new class names
        for item in self._annotation_items.values():
            # Force complete redraw of the item
            item.prepareGeometryChange()
            item.update(item.boundingRect())
    
    def set_show_class_names(self, show: bool):
        """Toggle between showing class names and IDs."""
        self._show_class_names = show
        
        # Update all existing annotation items to reflect the change
        for item in self._annotation_items.values():
            # Force complete redraw of the item including label
            item.prepareGeometryChange()
            item.update(item.boundingRect())
    
    def load_image(self, pixmap: QPixmap):
        """Load an image into the canvas."""
        # Clear previous
        self.clear_canvas()
        
        # Convert pixmap to ensure proper format for rendering
        # This helps with TIF files that might have unusual bit depths or formats
        if pixmap.depth() != 32:
            # Convert to 32-bit ARGB for consistent rendering
            image = pixmap.toImage()
            image = image.convertToFormat(QImage.Format.Format_ARGB32)
            pixmap = QPixmap.fromImage(image)
        
        # Add pixmap
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        
        # Fit the entire image in view while maintaining aspect ratio
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
    
    def set_annotations(self, annotations: List[Annotation]):
        """Set annotations."""
        self.clear_annotations()
        
        for ann in annotations:
            self.add_annotation(ann, emit_signal=False)
    
    def add_annotation(self, annotation: Annotation, emit_signal: bool = True):
        """Add an annotation."""
        if not annotation.id:
            annotation.id = f"ann_{len(self._annotation_items)}"
        
        # Create graphics item with reference to this canvas
        item = AnnotationItem(annotation, canvas=self)
        # Set the rect without moving position
        item.setRect(0, 0, annotation.rect.width(), annotation.rect.height())
        # Then set the position
        item.setPos(annotation.rect.x(), annotation.rect.y())
        self._scene.addItem(item)
        self._annotation_items[annotation.id] = item
        
        if emit_signal:
            self.annotationAdded.emit(annotation)
    
    def remove_annotation(self, annotation_id: str):
        """Remove an annotation."""
        if annotation_id in self._annotation_items:
            item = self._annotation_items[annotation_id]
            annotation = item.annotation
            
            # Temporarily disconnect to avoid selection changed on deleted item
            was_selected = item.isSelected()
            if was_selected:
                item.setSelected(False)
            
            self._scene.removeItem(item)
            del self._annotation_items[annotation_id]
            
            self.annotationDeleted.emit(annotation)
    
    def get_annotations(self) -> List[Annotation]:
        """Get all annotations."""
        return [item.annotation for item in self._annotation_items.values()]
    
    def get_selected_annotations(self) -> List[Annotation]:
        """Get selected annotations."""
        selected = []
        for item in self._annotation_items.values():
            try:
                if item.isSelected():
                    selected.append(item.annotation)
            except RuntimeError:
                # Item has been deleted, skip it
                pass
        return selected
    
    def update_annotation_items(self):
        """Update all annotation items to reflect changes."""
        for item in self._annotation_items.values():
            item._update_appearance()
            item.update()
    
    def clear_annotations(self):
        """Clear all annotations."""
        # Temporarily disconnect selection signal
        self._scene.selectionChanged.disconnect(self._on_scene_selection_changed)
        
        # Remove items
        for item in list(self._annotation_items.values()):
            self._scene.removeItem(item)
        self._annotation_items.clear()
        
        # Reconnect signal
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
    
    def clear_canvas(self):
        """Clear everything."""
        # Disconnect selection changed signal to prevent accessing deleted items
        self._scene.selectionChanged.disconnect(self._on_scene_selection_changed)
        
        # Clear items dictionary first to avoid accessing deleted items
        self._annotation_items.clear()
        
        # Now clear the scene
        self._scene.clear()
        self._pixmap_item = None
        
        # Reconnect the signal
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
    
    def select_all(self):
        """Select all annotations."""
        for item in self._annotation_items.values():
            item.setSelected(True)
    
    def delete_selected(self):
        """Delete selected annotations."""
        if self._is_read_only:
            return
        
        selected_ids = []
        for ann_id, item in self._annotation_items.items():
            if item.isSelected():
                selected_ids.append(ann_id)
        
        for ann_id in selected_ids:
            self.remove_annotation(ann_id)
    
    def undo(self):
        """Undo last action."""
        self._undo_stack.undo()
    
    def redo(self):
        """Redo last undone action."""
        self._undo_stack.redo()
    
    def _update_cursor(self):
        """Update cursor to crosshair for drawing."""
        self.setCursor(Qt.CursorShape.CrossCursor)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press with context-aware behavior."""
        if self._is_read_only:
            super().mousePressEvent(event)
            return
        
        scene_pos = self.mapToScene(event.pos())
        
        # Check for Ctrl+Click for panning
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Start panning
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Create a fake event without Ctrl to pass to parent
            fake_event = QMouseEvent(event.type(), event.position(), event.globalPosition(),
                                   Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                   Qt.KeyboardModifier.NoModifier)
            super().mousePressEvent(fake_event)
            return
        
        # Check for Middle Mouse Button for panning
        if event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Create a fake left-click event for panning
            fake_event = QMouseEvent(event.type(), event.position(), event.globalPosition(),
                                   Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                   Qt.KeyboardModifier.NoModifier)
            super().mousePressEvent(fake_event)
            return
        
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on an annotation
            item = self._scene.itemAt(scene_pos, self.transform())
            
            if isinstance(item, AnnotationItem):
                # Clicking on annotation - let it handle selection/movement
                super().mousePressEvent(event)
            elif self._pixmap_item and self._pixmap_item.contains(scene_pos):
                # Clicking on empty area - start drawing
                self._is_drawing = True
                self._draw_start_pos = scene_pos
                
                # Create preview rectangle with better visibility
                # Use a cyan color with thicker line for better contrast on all backgrounds
                preview_pen = QPen(QColor(0, 255, 255), 4, Qt.PenStyle.DashLine)
                preview_pen.setCosmetic(True)  # Keep consistent width regardless of zoom
                self._preview_rect = self._scene.addRect(
                    QRectF(scene_pos, scene_pos),
                    preview_pen,
                    QBrush(Qt.BrushStyle.NoBrush)
                )
                # Ensure it's on top
                self._preview_rect.setZValue(1000)
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        scene_pos = self.mapToScene(event.pos())
        
        if self._is_drawing and self._preview_rect:
            # Update preview rectangle
            rect = QRectF(self._draw_start_pos, scene_pos).normalized()
            self._preview_rect.setRect(rect)
            # Force scene update for better rendering with TIF files
            self._scene.update(rect.adjusted(-5, -5, 5, 5))
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        # Reset drag mode if we were panning
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        if self._is_drawing and self._preview_rect:
            # Finish drawing
            rect = self._preview_rect.rect()
            
            # Remove preview
            self._scene.removeItem(self._preview_rect)
            self._preview_rect = None
            
            # Create annotation if rect is valid
            if rect.width() > 10 and rect.height() > 10:
                annotation = Annotation(
                    class_id=self._current_class_id,
                    rect=rect
                )
                self.add_annotation(annotation)
            
            self._is_drawing = False
        
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        # Always zoom with mouse wheel
        delta = event.angleDelta().y()
        scale_factor = 1.1 if delta > 0 else 0.9
        
        # Get the position to zoom towards
        old_pos = self.mapToScene(event.position().toPoint())
        
        # Scale view
        self.scale(scale_factor, scale_factor)
        
        # Adjust position to zoom towards cursor
        new_pos = self.mapToScene(event.position().toPoint())
        delta_pos = new_pos - old_pos
        self.translate(delta_pos.x(), delta_pos.y())
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
        elif event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.select_all()
        elif event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undo()
        elif event.key() == Qt.Key.Key_Y and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.redo()
        # Handle number keys 0-9 for class selection when annotations are selected
        elif event.key() >= Qt.Key.Key_0 and event.key() <= Qt.Key.Key_9 and not event.modifiers():
            # Get the number pressed (0-9)
            class_id = event.key() - Qt.Key.Key_0
            
            # Check if we have selected annotations and this class ID exists
            selected_annotations = self.get_selected_annotations()
            if selected_annotations:
                # Check if this class_id is valid (exists in our class names)
                if class_id in self._class_names:
                    # Update all selected annotations to this class
                    for item in self._annotation_items.values():
                        if item.isSelected():
                            item.annotation.class_id = class_id
                            item._update_appearance()
                            item.update()  # Force redraw
                            self.annotationModified.emit(item.annotation)
        else:
            super().keyPressEvent(event)
    
    @pyqtSlot()
    def _on_scene_selection_changed(self):
        """Handle scene selection changes and emit our signal."""
        # Only emit if we have items (scene hasn't been cleared)
        if self._annotation_items:
            selected_annotations = self.get_selected_annotations()
            self.selectionChanged.emit(selected_annotations)