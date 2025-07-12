"""Utility to fix QComboBox dropdown hover issues."""

from PyQt6.QtWidgets import QComboBox, QListView, QListWidget
from PyQt6.QtCore import Qt


def fix_combobox_dropdown(combo: QComboBox) -> None:
    """
    Fix dropdown hover highlighting for a QComboBox by using a custom QListView.
    
    This ensures that dropdown items have proper hover highlighting like the
    "Filter by Category" dropdown in the Auto-Annotation tab.
    
    Args:
        combo: The QComboBox to fix
    """
    # Check if this combo already has a custom view (like the category filter)
    # Don't replace existing custom views to avoid breaking references
    current_view = combo.view()
    
    # Skip if it's already a QListWidget (custom view like category filter)
    if isinstance(current_view, QListWidget):
        return
    
    # Skip if the view already has custom styling
    if current_view and hasattr(current_view, 'styleSheet') and current_view.styleSheet():
        return
    
    # Create a custom list view
    list_view = QListView()
    
    # Apply the same styling that works for the category filter
    list_view.setStyleSheet("""
        QListView {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            color: #cccccc;
            outline: none;
        }
        QListView::item {
            padding: 5px;
            background-color: #3c3c3c;
            color: #cccccc;
        }
        QListView::item:hover {
            background-color: #555555;
            color: #ffffff;
        }
        QListView::item:selected {
            background-color: #0d7377;
            color: white;
        }
        QListView::item:selected:hover {
            background-color: #14919b;
            color: white;
        }
    """)
    
    # Set the custom view
    combo.setView(list_view)
    
    # Ensure the dropdown is wide enough
    combo.view().setMinimumWidth(combo.minimumSizeHint().width())


def fix_all_comboboxes_in_widget(widget) -> None:
    """
    Recursively find and fix all QComboBox widgets in a widget tree.
    
    Args:
        widget: The root widget to search in
    """
    # Check if the widget itself is a QComboBox
    if isinstance(widget, QComboBox):
        fix_combobox_dropdown(widget)
    
    # Recursively check all children
    for child in widget.findChildren(QComboBox):
        fix_combobox_dropdown(child)