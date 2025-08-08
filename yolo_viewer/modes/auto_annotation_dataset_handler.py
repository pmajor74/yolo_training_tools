"""Dataset management functionality for auto-annotation mode."""

import random
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from PyQt6.QtWidgets import QMessageBox, QFileDialog
from PyQt6.QtCore import QObject, pyqtSignal

from ..utils.yolo_format import parse_yolo_annotation
from .auto_annotation_data_classes import WorkflowState
from ..core.constants import IMAGE_EXTENSIONS


class DatasetHandler(QObject):
    """Handles dataset operations for auto-annotation mode."""
    
    # Signals
    datasetLoaded = pyqtSignal(Path, dict)  # yaml_path, data
    datasetCreated = pyqtSignal(Path)  # yaml_path
    splitCompleted = pyqtSignal(str, dict)  # message, split_counts
    error = pyqtSignal(str)  # error message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self._dataset_yaml_path: Optional[Path] = None
        self._dataset_class_names: Dict[int, str] = {}
        
    @property
    def dataset_yaml_path(self) -> Optional[Path]:
        """Get current dataset YAML path."""
        return self._dataset_yaml_path
        
    @property
    def dataset_class_names(self) -> Dict[int, str]:
        """Get current dataset class names."""
        return self._dataset_class_names
    
    def manage_dataset(self) -> Optional[Path]:
        """
        Manage dataset - load existing only.
        
        Returns:
            Path to dataset YAML if successful, None otherwise
        """
        # Load existing dataset only
        yaml_path, _ = QFileDialog.getOpenFileName(
            self._parent, "Select data.yaml", "", "YAML files (*.yaml *.yml)"
        )
        if yaml_path:
            return self.load_dataset_yaml(Path(yaml_path))
        
        return None
    
    def load_dataset_yaml(self, yaml_path: Path) -> Optional[Path]:
        """
        Load dataset from YAML file.
        
        Args:
            yaml_path: Path to data.yaml file
            
        Returns:
            Path to dataset YAML if successful, None otherwise
        """
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            self._dataset_yaml_path = yaml_path
            
            # Load class names if available
            if 'names' in data:
                self._dataset_class_names = data['names']
            
            self.datasetLoaded.emit(yaml_path, data)
            
            # Removed popup confirmation - dataset loading is now silent
            return yaml_path
            
        except Exception as e:
            self.error.emit(f"Failed to load dataset: {str(e)}")
            QMessageBox.critical(self._parent, "Error", f"Failed to load dataset: {str(e)}")
            return None
    
    def create_new_dataset(self, folder: Path) -> Optional[Path]:
        """
        Create new dataset structure.
        
        Args:
            folder: Folder where to create dataset
            
        Returns:
            Path to created dataset YAML if successful, None otherwise
        """
        try:
            # Create basic data.yaml
            yaml_path = folder / "data.yaml"
            data = {
                'path': '.',  # Current directory
                'train': 'train',
                'val': 'val',
                'test': 'test',
                'nc': 2,  # Default to 2 classes
                'names': {0: 'QR', 1: 'DATAMATRIX'}  # Default names
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            self._dataset_yaml_path = yaml_path
            self._dataset_class_names = data['names']
            
            self.datasetCreated.emit(yaml_path)
            
            QMessageBox.information(self._parent, "Dataset Created", 
                                  f"Dataset structure created at:\n{folder}")
            return yaml_path
            
        except Exception as e:
            self.error.emit(f"Failed to create dataset: {str(e)}")
            QMessageBox.critical(self._parent, "Error", f"Failed to create dataset: {str(e)}")
            return None
    
    def execute_dataset_split(self, current_folder: Path, train_pct: int, val_pct: int, 
                            test_pct: int, workflow_state: WorkflowState,
                            last_exported_paths: List[str] = None) -> bool:
        """
        Execute dataset split operation on the current folder.
        
        Args:
            current_folder: Folder containing images to split
            train_pct: Training split percentage
            val_pct: Validation split percentage  
            test_pct: Test split percentage
            workflow_state: Current workflow state
            last_exported_paths: List of recently exported image paths to clean up
            
        Returns:
            True if split was successful, False otherwise
        """
        if not current_folder or not self._dataset_yaml_path:
            QMessageBox.warning(self._parent, "Missing Requirements", 
                              "Please ensure you have a folder selected and dataset loaded.")
            return False
        
        # Convert percentages to fractions
        train_pct = train_pct / 100
        val_pct = val_pct / 100
        test_pct = test_pct / 100
        
        try:
            # Load data.yaml to get the paths
            with open(self._dataset_yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
            
            # Get dataset root from data.yaml location
            dataset_root = self._dataset_yaml_path.parent
            
            # Check if there's a 'path' field that specifies a base path
            base_path = data_yaml.get('path', '')
            if base_path:
                # If path is absolute, use it; otherwise, make it relative to dataset root
                if Path(base_path).is_absolute():
                    dataset_root = Path(base_path)
                else:
                    dataset_root = dataset_root / base_path
            
            # Extract paths from data.yaml
            train_path = data_yaml.get('train', 'train')
            val_path = data_yaml.get('val', 'val')
            test_path = data_yaml.get('test', 'test') if test_pct > 0 else None
            
            # Clean existing split directories first
            self._clean_split_directories(dataset_root, train_path, val_path, test_path)
            
            # Get ALL annotated images from the current folder
            annotated_images = self._get_annotated_images(current_folder)
            
            if not annotated_images:
                QMessageBox.warning(self._parent, "No Annotated Images", 
                                  "No images with annotations found in the current folder.")
                return False
            
            # Check if we have enough images for the split
            total_images = len(annotated_images)
            if total_images < 2:
                QMessageBox.warning(self._parent, "Insufficient Images", 
                                  f"Only {total_images} annotated image(s) found.\n"
                                  "At least 2 images are required for training with validation.")
                return False
            
            # Warn if split might result in empty validation set
            if not self._validate_split_sizes(total_images, val_pct):
                return False
            
            # Perform stratified splitting
            train_files, val_files, test_files = self._stratified_split(
                annotated_images, train_pct, val_pct, test_pct, total_images
            )
            
            # Create split directories and copy files
            copied_counts = self._copy_to_splits(
                dataset_root, train_path, val_path, test_path,
                train_files, val_files, test_files
            )
            
            # Emit completion signal
            split_message = (
                f"Dataset split complete:\n"
                f"Copied {total_images} annotated images from source folder\n\n"
                f"Train: {copied_counts['train']} images → {train_path}\n"
                f"Val: {copied_counts['val']} images → {val_path}\n"
                f"Test: {copied_counts['test']} images → {test_path if test_path else 'N/A'}\n\n"
                f"Dataset root: {dataset_root}\n\n"
                f"Files were COPIED (originals remain in source folder)."
            )
            
            self.splitCompleted.emit(split_message, copied_counts)
            
            # Only show popup if workflow is NOT enabled
            if not hasattr(self._parent, '_workflow_enabled') or not self._parent._workflow_enabled:
                QMessageBox.information(self._parent, "Split Complete", split_message)
            
            return True
                
        except Exception as e:
            self.error.emit(f"Failed to split dataset: {str(e)}")
            QMessageBox.critical(self._parent, "Split Error", f"Failed to split dataset: {str(e)}")
            return False
    
    def _clean_split_directories(self, dataset_root: Path, train_path: str, 
                               val_path: str, test_path: Optional[str]):
        """Clean existing split directories."""
        for split_name, split_path in [("train", train_path), ("val", val_path), ("test", test_path)]:
            if split_path:
                if Path(split_path).is_absolute():
                    split_dir = Path(split_path)
                else:
                    split_dir = dataset_root / split_path
                
                if split_dir.exists():
                    # Remove all files in the directory
                    for file in split_dir.iterdir():
                        if file.is_file():
                            file.unlink()
                    # Try to remove the directory itself
                    try:
                        split_dir.rmdir()
                    except:
                        pass  # Directory might have subdirectories
    
    def _get_annotated_images(self, folder: Path) -> List[Path]:
        """Get all annotated images from folder."""
        annotated_images = []
        
        for ext in IMAGE_EXTENSIONS:
            # Check lowercase extensions
            for img_path in folder.glob(f'*{ext}'):
                ann_path = img_path.with_suffix('.txt')
                if ann_path.exists():
                    annotated_images.append(img_path)
            # Check uppercase extensions
            for img_path in folder.glob(f'*{ext.upper()}'):
                ann_path = img_path.with_suffix('.txt')
                if ann_path.exists():
                    annotated_images.append(img_path)
        
        return annotated_images
    
    def _validate_split_sizes(self, total_images: int, val_pct: float) -> bool:
        """Validate that split sizes are reasonable."""
        min_val_images = int(total_images * val_pct)
        if min_val_images == 0 and val_pct > 0:
            reply = QMessageBox.question(
                self._parent, "Small Dataset Warning",
                f"With {total_images} images and {int(val_pct*100)}% validation split,\n"
                f"the validation set will be empty (needs at least {int(1/val_pct)} images).\n\n"
                "This will cause training to fail. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            return reply == QMessageBox.StandardButton.Yes
        return True
    
    def _stratified_split(self, annotated_images: List[Path], train_pct: float,
                         val_pct: float, test_pct: float, total_images: int) -> Tuple[List[Path], List[Path], List[Path]]:
        """Perform stratified split of images based on class distribution."""
        # Group images by their class combination
        images_by_class_combo = defaultdict(list)
        
        for img_path in annotated_images:
            ann_path = img_path.with_suffix('.txt')
            annotations = parse_yolo_annotation(ann_path)
            
            if annotations:
                # Get unique classes in this image
                classes_in_image = set()
                for ann in annotations:
                    class_id = int(ann[0])
                    classes_in_image.add(class_id)
                
                # Create a string key for this combination of classes
                combo_key = ",".join(str(c) for c in sorted(classes_in_image))
                images_by_class_combo[combo_key].append(img_path)
            else:
                # Images with empty annotations
                images_by_class_combo["empty"].append(img_path)
        
        # Now split each class combination proportionally
        train_files = []
        val_files = []
        test_files = []
        
        # Check if we need to ensure minimum validation set
        ensure_val_minimum = total_images >= 2 and val_pct > 0
        
        for combo_key, combo_images in images_by_class_combo.items():
            # Shuffle images for this combination
            random.shuffle(combo_images)
            
            # Calculate split sizes for this combination
            combo_total = len(combo_images)
            combo_train_size = int(combo_total * train_pct)
            combo_val_size = int(combo_total * val_pct)
            combo_test_size = int(combo_total * test_pct)
            
            # Handle special cases for small combos
            combo_train_size, combo_val_size, combo_test_size = self._adjust_split_sizes(
                combo_total, train_pct, val_pct, test_pct,
                combo_train_size, combo_val_size, combo_test_size,
                ensure_val_minimum, len(val_files)
            )
            
            # Assign to splits based on calculated sizes
            train_end = combo_train_size
            val_end = train_end + combo_val_size
            
            train_files.extend(combo_images[:train_end])
            val_files.extend(combo_images[train_end:val_end])
            # Only add to test if test ratio > 0
            if test_pct > 0 and combo_test_size > 0:
                test_files.extend(combo_images[val_end:val_end + combo_test_size])
        
        # Final check - ensure we have at least one image in train and val
        if val_pct > 0 and len(val_files) == 0 and len(train_files) > 1:
            # Move one image from train to val
            val_files.append(train_files.pop())
        
        return train_files, val_files, test_files
    
    def _adjust_split_sizes(self, combo_total: int, train_pct: float, val_pct: float, test_pct: float,
                          combo_train_size: int, combo_val_size: int, combo_test_size: int,
                          ensure_val_minimum: bool, current_val_count: int) -> Tuple[int, int, int]:
        """Adjust split sizes for edge cases."""
        if combo_total == 1:
            # Only 1 sample - assign based on highest ratio
            # But if we need val samples and don't have any yet, prioritize val
            if ensure_val_minimum and current_val_count == 0 and val_pct > 0:
                return 0, 1, 0
            elif train_pct >= val_pct and train_pct >= test_pct:
                return 1, 0, 0
            elif val_pct >= test_pct:
                return 0, 1, 0
            else:
                return 0, 0, 1
                
        elif combo_total == 2:
            # 2 samples - split between two largest ratios
            # But ensure at least one goes to val if needed
            if ensure_val_minimum and current_val_count == 0 and val_pct > 0:
                return 1, 1, 0
            else:
                ratios = [("train", train_pct), ("val", val_pct), ("test", test_pct)]
                ratios.sort(key=lambda x: x[1], reverse=True)
                
                # Reset sizes
                train_size, val_size, test_size = 0, 0, 0
                
                # Assign 1 to each of the top 2 ratios (if > 0)
                assigned = 0
                for split_name, ratio in ratios:
                    if ratio > 0 and assigned < 2:
                        if split_name == "train":
                            train_size = 1
                        elif split_name == "val":
                            val_size = 1
                        elif split_name == "test":
                            test_size = 1
                        assigned += 1
                
                return train_size, val_size, test_size
                
        else:
            # 3 or more samples - ensure at least one for validation if needed
            if ensure_val_minimum and current_val_count == 0 and combo_val_size == 0 and val_pct > 0:
                combo_val_size = 1
                combo_train_size = max(0, combo_train_size - 1)
            
            # Ensure sizes sum to combo_total
            current_sum = combo_train_size + combo_val_size + combo_test_size
            if current_sum < combo_total:
                diff = combo_total - current_sum
                # Add remainder based on ratio priority
                if train_pct > 0:
                    combo_train_size += diff
                elif val_pct > 0:
                    combo_val_size += diff
                elif test_pct > 0:
                    combo_test_size += diff
            
            return combo_train_size, combo_val_size, combo_test_size
    
    def _copy_to_splits(self, dataset_root: Path, train_path: str, val_path: str,
                       test_path: Optional[str], train_files: List[Path],
                       val_files: List[Path], test_files: List[Path]) -> Dict[str, int]:
        """Copy files to split directories."""
        splits = [
            ("train", train_path, train_files),
            ("val", val_path, val_files)
        ]
        if test_path and test_files:
            splits.append(("test", test_path, test_files))
        
        # Track actual copied counts
        copied_counts = {"train": 0, "val": 0, "test": 0}
        
        for split_name, split_path, files in splits:
            if files:
                # Handle absolute vs relative paths
                if Path(split_path).is_absolute():
                    split_dir = Path(split_path)
                else:
                    split_dir = dataset_root / split_path
                
                # Create the directory (images and labels go in the same folder)
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy files (not move - keep originals in place)
                for img_file in files:
                    try:
                        # Copy image
                        shutil.copy2(str(img_file), str(split_dir / img_file.name))
                        copied_counts[split_name] += 1
                        
                        # Copy corresponding label
                        label_file = img_file.with_suffix('.txt')
                        if label_file.exists():
                            shutil.copy2(str(label_file), str(split_dir / label_file.name))
                    except Exception as e:
                        print(f"Error copying {img_file}: {e}")
                        continue
        
        return copied_counts