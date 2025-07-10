"""Dataset configuration dialog for creating/editing YOLO data.yaml files."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QListWidget,
    QListWidgetItem, QSpinBox, QMessageBox, QFileDialog,
    QGridLayout, QWidget, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon

import yaml


class ClassItem(QListWidgetItem):
    """Custom list item for displaying classes."""
    
    def __init__(self, class_id: int, class_name: str):
        super().__init__()
        self.class_id = class_id
        self.class_name = class_name
        self.setText(f"{class_id}: {class_name}")
        
    def update_display(self):
        """Update the display text."""
        self.setText(f"{self.class_id}: {self.class_name}")


class DatasetConfigDialog(QDialog):
    """Dialog for configuring YOLO dataset structure and data.yaml."""
    
    # Signal emitted when dataset is saved
    datasetSaved = pyqtSignal(Path)
    
    def __init__(self, parent=None, yaml_path: Optional[Path] = None):
        super().__init__(parent)
        self.yaml_path = yaml_path
        self.is_editing = yaml_path is not None
        
        # Initialize data
        self.root_folder = ""
        self.train_folder = "train"
        self.val_folder = "val"
        self.test_folder = "test"
        self.classes: Dict[int, str] = {}
        
        # Load existing data if editing
        if self.is_editing and yaml_path.exists():
            self._load_existing_yaml()
        elif not self.is_editing:
            # For new datasets, start with empty classes
            self.classes = {}
        
        self._setup_ui()
        self._apply_dark_theme()
        
        # Update class list with initial classes
        if self.classes:
            self._update_class_list()
        
    def _setup_ui(self):
        """Setup the dialog UI."""
        title = "Edit Dataset Configuration" if self.is_editing else "Create New Dataset"
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(700, 600)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Folder Configuration
        folder_group = QGroupBox("Dataset Folders")
        folder_layout = QGridLayout(folder_group)
        folder_layout.setSpacing(8)
        
        row = 0
        
        # Save location (only for new datasets)
        if not self.yaml_path:
            save_label = QLabel("Save Location:")
            save_label.setStyleSheet("color: #14ffec; font-weight: bold;")  # Highlight this field
            folder_layout.addWidget(save_label, row, 0)
            self.save_location_edit = QLineEdit()
            self.save_location_edit.setPlaceholderText("Where to save data.yaml (e.g., /path/to/dataset/data.yaml)")
            self.save_location_edit.setStyleSheet("border: 1px solid #14ffec;")  # Highlight the input
            folder_layout.addWidget(self.save_location_edit, row, 1)
            
            self.save_btn = QPushButton("Browse...")
            self.save_btn.clicked.connect(self._browse_save_location)
            folder_layout.addWidget(self.save_btn, row, 2)
            row += 1
        
        # Root folder
        folder_layout.addWidget(QLabel("Root Folder:"), row, 0)
        self.root_edit = QLineEdit(self.root_folder)
        self.root_edit.setPlaceholderText("Select dataset root folder...")
        folder_layout.addWidget(self.root_edit, row, 1)
        
        self.root_btn = QPushButton("Browse...")
        self.root_btn.clicked.connect(self._browse_root_folder)
        folder_layout.addWidget(self.root_btn, row, 2)
        row += 1
        
        # Train folder
        folder_layout.addWidget(QLabel("Train Folder:"), row, 0)
        self.train_edit = QLineEdit(self.train_folder)
        self.train_edit.setPlaceholderText("e.g., train/images or train")
        folder_layout.addWidget(self.train_edit, row, 1)
        row += 1
        
        # Validation folder
        folder_layout.addWidget(QLabel("Val Folder:"), row, 0)
        self.val_edit = QLineEdit(self.val_folder)
        self.val_edit.setPlaceholderText("e.g., val/images or val")
        folder_layout.addWidget(self.val_edit, row, 1)
        row += 1
        
        # Test folder
        folder_layout.addWidget(QLabel("Test Folder:"), row, 0)
        self.test_edit = QLineEdit(self.test_folder)
        self.test_edit.setPlaceholderText("e.g., test/images or test (optional)")
        folder_layout.addWidget(self.test_edit, row, 1)
        row += 1
        
        # Info label
        info_text = "💡 Tips:\n"
        info_text += "• Standard YOLO structure: train/images, val/images, test/images\n"
        info_text += "• If data.yaml is saved in the dataset root (same folder as train/val/test), leave Root Folder empty\n"
        info_text += "• If data.yaml is saved elsewhere, set Root Folder to point to your dataset directory"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #888888; font-size: 11px;")
        info_label.setWordWrap(True)
        folder_layout.addWidget(info_label, row, 0, 1, 3)
        
        layout.addWidget(folder_group)
        
        # Class Configuration
        class_group = QGroupBox("Classes")
        class_layout = QVBoxLayout(class_group)
        
        # Class list
        self.class_list = QListWidget()
        self.class_list.setSortingEnabled(False)
        self.class_list.itemDoubleClicked.connect(self._edit_class)
        class_layout.addWidget(self.class_list)
        
        # Class controls
        class_controls = QHBoxLayout()
        class_controls.setSpacing(5)
        
        self.add_class_btn = QPushButton("➕ Add Class")
        self.add_class_btn.clicked.connect(self._add_class)
        class_controls.addWidget(self.add_class_btn)
        
        self.edit_class_btn = QPushButton("✏️ Edit Class")
        self.edit_class_btn.clicked.connect(self._edit_selected_class)
        self.edit_class_btn.setEnabled(False)
        class_controls.addWidget(self.edit_class_btn)
        
        self.remove_class_btn = QPushButton("🗑️ Remove Class")
        self.remove_class_btn.clicked.connect(self._remove_class)
        self.remove_class_btn.setEnabled(False)
        class_controls.addWidget(self.remove_class_btn)
        
        class_controls.addStretch()
        
        class_layout.addLayout(class_controls)
        
        # Enable/disable buttons based on selection
        self.class_list.itemSelectionChanged.connect(self._update_class_buttons)
        
        layout.addWidget(class_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._save_dataset)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
        
        # Populate class list
        self._update_class_list()
        
    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLineEdit {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QLineEdit:focus {
                border-color: #0d7377;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5d61;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #0d7377;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
            QSpinBox {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
        """)
        
    def _load_existing_yaml(self):
        """Load existing data.yaml file."""
        try:
            with open(self.yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Get root folder (path field or parent directory)
            if 'path' in data:
                self.root_folder = str(data['path'])
            else:
                self.root_folder = str(self.yaml_path.parent)
            
            # Get split folders
            self.train_folder = data.get('train', 'train')
            self.val_folder = data.get('val', 'val')
            self.test_folder = data.get('test', 'test')
            
            # Get classes
            names = data.get('names', {})
            if isinstance(names, dict):
                self.classes = {int(k): v for k, v in names.items()}
            elif isinstance(names, list):
                self.classes = {i: name for i, name in enumerate(names)}
                
        except Exception as e:
            QMessageBox.warning(
                self, "Load Error",
                f"Failed to load existing data.yaml:\n{str(e)}"
            )
            
    def _browse_root_folder(self):
        """Browse for dataset root folder."""
        # Start from current root or parent of yaml if editing
        start_dir = self.root_folder
        if not start_dir and self.yaml_path:
            start_dir = str(self.yaml_path.parent)
        
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Root Folder",
            start_dir
        )
        
        if folder:
            self.root_edit.setText(folder)
            
    def _browse_save_location(self):
        """Browse for save location (new datasets only)."""
        # Default to root folder if specified
        start_dir = self.root_edit.text() or str(Path.cwd())
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Dataset Configuration",
            str(Path(start_dir) / "data.yaml"),
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )
        
        if file_path:
            self.save_location_edit.setText(file_path)
            
    def _update_class_list(self):
        """Update the class list widget."""
        self.class_list.clear()
        
        # Sort by class ID
        sorted_classes = sorted(self.classes.items())
        
        for class_id, class_name in sorted_classes:
            item = ClassItem(class_id, class_name)
            self.class_list.addItem(item)
            
    def _update_class_buttons(self):
        """Update class button states based on selection."""
        has_selection = len(self.class_list.selectedItems()) > 0
        self.edit_class_btn.setEnabled(has_selection)
        self.remove_class_btn.setEnabled(has_selection)
        
    def _add_class(self):
        """Add a new class."""
        # Find next available ID
        next_id = 0
        if self.classes:
            next_id = max(self.classes.keys()) + 1
        
        # Get class name from user
        from PyQt6.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(
            self, "Add Class",
            f"Enter name for class {next_id}:",
            text=""
        )
        
        if ok and name:
            self.classes[next_id] = name
            self._update_class_list()
            
    def _edit_selected_class(self):
        """Edit the selected class."""
        items = self.class_list.selectedItems()
        if items:
            self._edit_class(items[0])
            
    def _edit_class(self, item: ClassItem):
        """Edit a class."""
        from PyQt6.QtWidgets import QInputDialog
        
        # Get new name
        name, ok = QInputDialog.getText(
            self, "Edit Class",
            f"Enter new name for class {item.class_id}:",
            text=item.class_name
        )
        
        if ok and name:
            self.classes[item.class_id] = name
            item.class_name = name
            item.update_display()
            
    def _remove_class(self):
        """Remove selected class."""
        items = self.class_list.selectedItems()
        if not items:
            return
            
        item = items[0]
        if isinstance(item, ClassItem):
            reply = QMessageBox.question(
                self, "Remove Class",
                f"Remove class {item.class_id}: {item.class_name}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                del self.classes[item.class_id]
                self._update_class_list()
                
    def _validate_configuration(self) -> Tuple[bool, str]:
        """Validate the dataset configuration."""
        # Check if we have classes
        if not self.classes:
            return False, "At least one class must be defined"
        
        # Check folder paths
        train = self.train_edit.text().strip()
        val = self.val_edit.text().strip()
        
        if not train:
            return False, "Train folder path is required"
        if not val:
            return False, "Validation folder path is required"
        
        # If creating new dataset, check if data.yaml location is specified
        if not self.is_editing and hasattr(self, 'save_location_edit'):
            save_location = self.save_location_edit.text().strip()
            if not save_location:
                return False, "Please specify where to save data.yaml"
        
        return True, ""
        
    def _save_dataset(self):
        """Save the dataset configuration."""
        # Validate
        valid, message = self._validate_configuration()
        if not valid:
            QMessageBox.warning(self, "Validation Error", message)
            return
        
        # Get save location
        save_path = self.yaml_path
        if not save_path:
            # For new datasets, use the save location field
            if hasattr(self, 'save_location_edit'):
                save_location = self.save_location_edit.text().strip()
                save_path = Path(save_location)
            else:
                # Fallback to file dialog
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "Save data.yaml",
                    "data.yaml",
                    "YAML files (*.yaml *.yml)"
                )
                
                if not save_path:
                    return
                    
                save_path = Path(save_path)
        
        # Build data dictionary
        data = {
            'names': self.classes,
            'nc': len(self.classes),
            'train': self.train_edit.text().strip(),
            'val': self.val_edit.text().strip()
        }
        
        # Add test if specified
        test = self.test_edit.text().strip()
        if test:
            data['test'] = test
        
        # Handle root folder and adjust paths accordingly
        root = self.root_edit.text().strip()
        save_path_parent = save_path.parent.resolve()
        
        if root:
            # Root folder is specified
            root_path = Path(root).resolve()
            
            if root_path != save_path_parent:
                # data.yaml is in a different location than dataset root
                # Add 'path' field pointing to the dataset root
                try:
                    rel_path = os.path.relpath(root_path, save_path_parent)
                    data['path'] = rel_path
                except ValueError:
                    # Can't make relative path (different drives on Windows)
                    data['path'] = str(root_path)
            # else: data.yaml is in the same folder as dataset root, no 'path' needed
        else:
            # No root folder specified - assume data.yaml is in the dataset root
            # This is the case when data.yaml is saved alongside train/val/test folders
            # No 'path' field needed, train/val/test paths are relative to data.yaml
            pass
        
        # Save to file
        try:
            with open(save_path, 'w') as f:
                # Write header comments
                f.write("# YOLO Dataset Configuration\n")
                if 'path' in data:
                    f.write("# Dataset root directory (relative to this file or absolute path)\n")
                else:
                    f.write("# Dataset paths are relative to this file's location\n")
                f.write("\n")
                
                # Write the YAML data
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
            self.yaml_path = save_path
            self.datasetSaved.emit(save_path)
            self.accept()
            
            QMessageBox.information(
                self, "Success",
                f"Dataset configuration saved to:\n{save_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Save Error",
                f"Failed to save data.yaml:\n{str(e)}"
            )