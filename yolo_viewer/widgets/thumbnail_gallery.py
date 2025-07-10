"""Enhanced thumbnail gallery with annotations and size control."""

from typing import List, Optional, Dict, Tuple, Set
from pathlib import Path
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QListView, QStyledItemDelegate, QSlider, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QStyle
)
from PyQt6.QtCore import (
    Qt, QAbstractListModel, QModelIndex, QSize,
    pyqtSignal, QRect, QRectF, QTimer, QEvent
)
from PyQt6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QBrush, QFont,
    QFontMetrics
)

from ..core.constants import ANNOTATION_COLORS


@dataclass
class ThumbnailData:
    """Data for a thumbnail."""
    image_path: str
    pixmap: Optional[QPixmap] = None
    annotations: List[Tuple[int, float, float, float, float, float]] = None  # class_id, x, y, w, h, conf
    is_modified: bool = False
    
    def __post_init__(self):
        if self.annotations is None:
            self.annotations = []


class ThumbnailModel(QAbstractListModel):
    """Model for enhanced thumbnail gallery."""
    
    def __init__(self):
        super().__init__()
        self._items: List[ThumbnailData] = []
        self._thumbnail_size = 150
    
    def set_items(self, items: List[ThumbnailData]):
        """Set all items."""
        self.beginResetModel()
        self._items = items
        self.endResetModel()
    
    def add_items(self, items: List[ThumbnailData]):
        """Add items without resetting the entire model."""
        if not items:
            return
        start_row = len(self._items)
        end_row = start_row + len(items) - 1
        self.beginInsertRows(QModelIndex(), start_row, end_row)
        self._items.extend(items)
        self.endInsertRows()
    
    def set_thumbnail_size(self, size: int):
        """Set thumbnail size."""
        self._thumbnail_size = size
        # Force all items to update
        if self._items:
            self.dataChanged.emit(
                self.index(0), 
                self.index(len(self._items) - 1)
            )
    
    def update_item_annotations(self, image_path: str, annotations: List[Tuple], is_modified: bool = True):
        """Update annotations for an item."""
        for i, item in enumerate(self._items):
            if item.image_path == image_path:
                item.annotations = annotations
                item.is_modified = is_modified
                index = self.index(i)
                self.dataChanged.emit(index, index)
                break
    
    def get_item(self, index: int) -> Optional[ThumbnailData]:
        """Get item at index."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None
    
    def remove_items_by_paths(self, paths: List[str]):
        """Remove items with the specified image paths."""
        if not paths:
            return
        
        paths_set = set(paths)
        # Find indices to remove (in reverse order to avoid index shifting)
        indices_to_remove = []
        for i, item in enumerate(self._items):
            if item.image_path in paths_set:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain valid indices
        for index in reversed(sorted(indices_to_remove)):
            self.beginRemoveRows(QModelIndex(), index, index)
            self._items.pop(index)
            self.endRemoveRows()
    
    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._items)
    
    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        item = self._items[index.row()]
        
        if role == Qt.ItemDataRole.DisplayRole:
            return Path(item.image_path).name
        elif role == Qt.ItemDataRole.UserRole:
            return item
        elif role == Qt.ItemDataRole.SizeHintRole:
            return QSize(self._thumbnail_size + 10, self._thumbnail_size + 30)
        
        return None


class ThumbnailDelegate(QStyledItemDelegate):
    """Delegate for rendering thumbnails with annotations."""
    
    def paint(self, painter: QPainter, option, index: QModelIndex):
        """Paint thumbnail with annotations."""
        # Ensure painter is active before painting
        if not painter.isActive():
            return
            
        painter.save()
        
        item = index.data(Qt.ItemDataRole.UserRole)
        if not item:
            painter.restore()
            return
        
        rect = option.rect
        
        # Draw selection
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(rect, QColor(255, 215, 0, 50))
            painter.setPen(QPen(QColor(255, 215, 0), 3))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))
        
        # Calculate image rect
        margin = 5
        img_rect = rect.adjusted(margin, margin, -margin, -margin - 20)
        
        # Draw thumbnail
        if item.pixmap and not item.pixmap.isNull() and img_rect.width() > 0 and img_rect.height() > 0:
            scaled_pixmap = item.pixmap.scaled(
                img_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Center the pixmap
            x = img_rect.x() + (img_rect.width() - scaled_pixmap.width()) // 2
            y = img_rect.y() + (img_rect.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
            
            # Draw annotations
            if item.annotations:
                self._draw_annotations(painter, QRect(x, y, scaled_pixmap.width(), scaled_pixmap.height()), 
                                     item.annotations, item.pixmap.size())
        else:
            # Draw placeholder
            painter.fillRect(img_rect, QColor(200, 200, 200))
            painter.drawText(img_rect, Qt.AlignmentFlag.AlignCenter, "Loading...")
        
        # Draw modified indicator
        if item.is_modified:
            indicator_rect = QRect(rect.right() - 20, rect.top() + 5, 15, 15)
            painter.fillRect(indicator_rect, QColor(255, 140, 0))
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(indicator_rect, Qt.AlignmentFlag.AlignCenter, "M")
        
        # Draw filename
        text_rect = QRect(rect.x(), rect.bottom() - 20, rect.width(), 20)
        painter.fillRect(text_rect, QColor(255, 255, 255, 200))
        painter.setPen(Qt.GlobalColor.black)
        painter.setFont(QFont("Arial", 9))
        
        metrics = QFontMetrics(painter.font())
        filename = Path(item.image_path).name
        elided_text = metrics.elidedText(filename, Qt.TextElideMode.ElideRight, rect.width() - 10)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, elided_text)
        
        painter.restore()
    
    def _draw_annotations(self, painter: QPainter, img_rect: QRect, annotations: List[Tuple], original_size: QSize):
        """Draw annotations on thumbnail."""
        # Calculate scale
        scale_x = img_rect.width() / original_size.width()
        scale_y = img_rect.height() / original_size.height()
        
        for ann in annotations:
            if len(ann) >= 5:
                class_id, x_center, y_center, width, height = ann[:5]
                
                # Convert normalized to pixel coordinates on thumbnail
                x = int((x_center - width/2) * img_rect.width() + img_rect.x())
                y = int((y_center - height/2) * img_rect.height() + img_rect.y())
                w = int(width * img_rect.width())
                h = int(height * img_rect.height())
                
                # Get color
                color = QColor(*ANNOTATION_COLORS[int(class_id) % len(ANNOTATION_COLORS)])
                painter.setPen(QPen(color, 2))
                painter.drawRect(x, y, w, h)


class ThumbnailGallery(QWidget):
    """Enhanced thumbnail gallery with size control."""
    
    # Signals
    imageSelected = pyqtSignal(str)
    imageDoubleClicked = pyqtSignal(str)  # emitted when image is double-clicked
    thumbnailSizeChanged = pyqtSignal(int)
    selectionChanged = pyqtSignal()  # emitted when selection changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Make widget focusable
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Setup UI
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Size control
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(50, 300)
        self.size_slider.setValue(150)
        self.size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.size_slider.setTickInterval(50)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.size_slider)
        
        self.size_label = QLabel("150px")
        size_layout.addWidget(self.size_label)
        
        layout.addLayout(size_layout)
        
        # List view
        self.list_view = QListView()
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setSpacing(5)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.list_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Set initial size
        initial_size = 150
        self.list_view.setIconSize(QSize(initial_size, initial_size))
        self.list_view.setGridSize(QSize(initial_size + 20, initial_size + 40))
        
        # Model and delegate
        self._model = ThumbnailModel()
        self._delegate = ThumbnailDelegate()
        
        # Set model first, then delegate
        self.list_view.setModel(self._model)
        # Defer delegate assignment to avoid early painting
        QTimer.singleShot(0, lambda: self.list_view.setItemDelegate(self._delegate))
        
        # Connect signals
        self.list_view.clicked.connect(self._on_clicked)
        self.list_view.doubleClicked.connect(self._on_double_clicked)
        self.list_view.selectionModel().selectionChanged.connect(
            lambda: self.selectionChanged.emit()
        )
        
        # Install event filter to catch keyboard events on list view
        self.list_view.installEventFilter(self)
        # Install on the viewport too - this is where scroll events actually happen
        self.list_view.viewport().installEventFilter(self)
        
        # Also install event filter on the gallery widget itself for wheel events
        self.installEventFilter(self)
        # Install on size slider too
        self.size_slider.installEventFilter(self)
        self.size_label.installEventFilter(self)
        
        layout.addWidget(self.list_view)
        
        # Cache for loaded images
        self._image_cache: Dict[str, QPixmap] = {}
    
    def load_images(self, image_paths: List[str], annotations_dict: Optional[Dict[str, List[Tuple]]] = None):
        """Load images with optional annotations."""
        # Clear any existing selection
        self.list_view.clearSelection()
        
        items = []
        seen_paths = set()
        
        for path in image_paths:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            # Load pixmap (with caching)
            pixmap = self._load_pixmap(path)
            
            # Get annotations if provided
            annotations = annotations_dict.get(path, []) if annotations_dict else []
            
            item = ThumbnailData(
                image_path=path,
                pixmap=pixmap,
                annotations=annotations,
                is_modified=False
            )
            items.append(item)
        
        self._model.set_items(items)
    
    def update_items_filter(self, visible_paths: Set[str]):
        """Update which items are visible without reloading."""
        # This would require implementing a QSortFilterProxyModel
        # For now, we'll use the existing load_images method
        # but optimize the auto-annotation mode to avoid repeated loads
        pass
    
    def update_image_annotations(self, image_path: str, annotations: List[Tuple], is_modified: bool = True):
        """Update annotations for an image."""
        self._model.update_item_annotations(image_path, annotations, is_modified)
    
    def get_image_annotations(self, image_path: str) -> Optional[List[Tuple]]:
        """Get annotations for a specific image."""
        for i in range(self._model.rowCount()):
            item = self._model.get_item(i)
            if item and item.image_path == image_path:
                return item.annotations
        return None
    
    def get_selected_paths(self) -> List[str]:
        """Get selected image paths."""
        paths = []
        for index in self.list_view.selectedIndexes():
            item = self._model.get_item(index.row())
            if item:
                paths.append(item.image_path)
        return paths
    
    def select_all(self):
        """Select all visible items in the gallery."""
        self.list_view.selectAll()
    
    def clear_selection(self):
        """Clear all selections in the gallery."""
        self.list_view.clearSelection()
    
    def _load_pixmap(self, path: str) -> Optional[QPixmap]:
        """Load pixmap with caching."""
        if path in self._image_cache:
            return self._image_cache[path]
        
        # Check if file exists before trying to load
        if not Path(path).exists():
            return None
            
        # Only create QPixmap if file exists
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self._image_cache[path] = pixmap
            return pixmap
        return None
    
    def _on_clicked(self, index: QModelIndex):
        """Handle click."""
        item = self._model.get_item(index.row())
        if item:
            self.imageSelected.emit(item.image_path)
    
    def _on_double_clicked(self, index: QModelIndex):
        """Handle double-click."""
        item = self._model.get_item(index.row())
        if item:
            self.imageDoubleClicked.emit(item.image_path)
    
    def _on_size_changed(self, value: int):
        """Handle size slider change."""
        self._model.set_thumbnail_size(value)
        self.size_label.setText(f"{value}px")
        
        # Update the list view's grid size
        self.list_view.setIconSize(QSize(value, value))
        self.list_view.setGridSize(QSize(value + 20, value + 40))
        
        # Force the view to update
        self.list_view.reset()
        
        self.thumbnailSizeChanged.emit(value)
    
    def get_selected_images(self) -> List[str]:
        """Get all selected image paths."""
        paths = []
        for index in self.list_view.selectionModel().selectedIndexes():
            item = self._model.get_item(index.row())
            if item:
                paths.append(item.image_path)
        return paths
    
    def select_all(self):
        """Select all images."""
        self.list_view.selectAll()
    
    def clear_selection(self):
        """Clear selection."""
        self.list_view.clearSelection()
    
    def setSelectionMode(self, mode):
        """Set selection mode."""
        self.list_view.setSelectionMode(mode)
    
    def keyPressEvent(self, event):
        """Handle key press events for navigation."""
        from PyQt6.QtCore import Qt, QItemSelectionModel, QItemSelection
        
        current_index = self.list_view.currentIndex()
        if not current_index.isValid():
            # No current selection, select first item
            if self._model.rowCount() > 0:
                self.list_view.setCurrentIndex(self._model.index(0))
                item = self._model.get_item(0)
                if item:
                    self.imageSelected.emit(item.image_path)
            return
        
        current_row = current_index.row()
        total_rows = self._model.rowCount()
        is_shift_pressed = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        
        # Store the anchor point for shift selection
        if not hasattr(self, '_selection_anchor'):
            self._selection_anchor = current_row
        
        # If shift is not pressed, update anchor to current position
        if not is_shift_pressed:
            self._selection_anchor = current_row
        
        # Helper function to handle navigation with optional shift selection
        def navigate_to_row(new_row):
            if 0 <= new_row < total_rows:
                new_index = self._model.index(new_row)
                selection_model = self.list_view.selectionModel()
                
                if is_shift_pressed:
                    # Clear current selection and select range from anchor to new position
                    selection_model.clearSelection()
                    
                    # Determine the range to select
                    start_row = min(self._selection_anchor, new_row)
                    end_row = max(self._selection_anchor, new_row)
                    
                    # Create selection for the range
                    selection = QItemSelection()
                    top_left = self._model.index(start_row)
                    bottom_right = self._model.index(end_row)
                    selection.select(top_left, bottom_right)
                    
                    # Apply the selection
                    selection_model.select(selection, QItemSelectionModel.SelectionFlag.Select)
                    
                    # Set current index without clearing selection
                    selection_model.setCurrentIndex(new_index, QItemSelectionModel.SelectionFlag.NoUpdate)
                else:
                    # Normal navigation - clear selection and move to new item
                    self.list_view.setCurrentIndex(new_index)
                    # Update anchor for future shift selections
                    self._selection_anchor = new_row
                
                # Emit signal for the current item
                item = self._model.get_item(new_row)
                if item:
                    self.imageSelected.emit(item.image_path)
                
                # Ensure the new item is visible
                self.list_view.scrollTo(new_index)
        
        # Calculate grid dimensions for up/down navigation
        view_width = self.list_view.viewport().width()
        grid_size = self.list_view.gridSize()
        columns = max(1, view_width // grid_size.width())
        
        if event.key() == Qt.Key.Key_Right:
            # Move to next image
            if is_shift_pressed:
                # For shift+right, just go to next item (no wrapping)
                next_row = min(current_row + 1, total_rows - 1)
            else:
                # Normal right with wrap-around
                next_row = (current_row + 1) % total_rows
            navigate_to_row(next_row)
                
        elif event.key() == Qt.Key.Key_Left:
            # Move to previous image
            if is_shift_pressed:
                # For shift+left, just go to previous item (no wrapping)
                prev_row = max(current_row - 1, 0)
            else:
                # Normal left with wrap-around
                prev_row = (current_row - 1) % total_rows
            navigate_to_row(prev_row)
            
        elif event.key() == Qt.Key.Key_Down:
            # Move down in the grid
            next_row = min(current_row + columns, total_rows - 1)
            navigate_to_row(next_row)
            
        elif event.key() == Qt.Key.Key_Up:
            # Move up in the grid
            prev_row = max(current_row - columns, 0)
            navigate_to_row(prev_row)
                
        elif event.key() == Qt.Key.Key_PageUp:
            # Scroll up by one page
            scroll_bar = self.list_view.verticalScrollBar()
            page_step = scroll_bar.pageStep()
            scroll_bar.setValue(scroll_bar.value() - page_step)
            
        elif event.key() == Qt.Key.Key_PageDown:
            # Scroll down by one page
            scroll_bar = self.list_view.verticalScrollBar()
            page_step = scroll_bar.pageStep()
            scroll_bar.setValue(scroll_bar.value() + page_step)
                
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)
    
    def remove_selected_items(self):
        """Remove currently selected items from the gallery."""
        selected_paths = self.get_selected_paths()
        if selected_paths:
            self._model.remove_items_by_paths(selected_paths)
            # Also remove from cache to free memory
            for path in selected_paths:
                if path in self._image_cache:
                    del self._image_cache[path]
    
    def eventFilter(self, obj, event):
        """Filter events from the widget and list view."""
        # Handle wheel events - prioritize list view to catch before scrolling
        if event.type() == QEvent.Type.Wheel:
            # Check for CTRL modifier first
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Handle zoom regardless of which child widget gets the event
                self._handle_wheel_event(event)
                return True  # Consume event to prevent any scrolling
        
        # Handle keyboard events specifically on the list view
        if obj == self.list_view and event.type() == QEvent.Type.KeyPress:
            self.keyPressEvent(event)
            return True
            
        return super().eventFilter(obj, event)
    
    def _handle_wheel_event(self, event):
        """Handle wheel events for thumbnail size adjustment."""
        # Check if CTRL is pressed
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Get current size
            current_size = self.size_slider.value()
            
            # Determine scroll direction
            delta = event.angleDelta().y()
            if delta > 0:
                # Scroll up - increase size
                new_size = min(current_size + 20, self.size_slider.maximum())
            else:
                # Scroll down - decrease size
                new_size = max(current_size - 20, self.size_slider.minimum())
            
            # Update slider (which will trigger size change)
            self.size_slider.setValue(new_size)
            
            # Consume the event
            return True
        
        # Let normal wheel events pass through
        return False
    
    def wheelEvent(self, event):
        """Override wheel event to handle CTRL+wheel directly."""
        # Check if CTRL is pressed
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Handle zoom
            self._handle_wheel_event(event)
            event.accept()  # Consume the event to prevent scrolling
        else:
            # Let normal scrolling happen
            super().wheelEvent(event)


# For backwards compatibility
EnhancedThumbnailGallery = ThumbnailGallery