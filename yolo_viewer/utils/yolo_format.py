"""YOLO format utilities for parsing and saving annotations."""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml
from ..core.constants import IMAGE_EXTENSIONS


def parse_yolo_annotation(file_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse YOLO format annotation file.
    
    Args:
        file_path: Path to annotation file
        
    Returns:
        List of annotations as (class_id, x_center, y_center, width, height)
    """
    annotations = []
    
    if not file_path.exists():
        return annotations
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append((class_id, x_center, y_center, width, height))
    except Exception as e:
        print(f"Error parsing annotation file {file_path}: {e}")
    
    return annotations


def save_yolo_annotation(file_path: Path, annotations: List[Tuple[int, float, float, float, float]]):
    """
    Save annotations in YOLO format.
    
    Args:
        file_path: Path to save annotation file
        annotations: List of annotations as (class_id, x_center, y_center, width, height)
    """
    try:
        with open(file_path, 'w') as f:
            for ann in annotations:
                class_id, x_center, y_center, width, height = ann[:5]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    except Exception as e:
        print(f"Error saving annotation file {file_path}: {e}")


def load_data_yaml(yaml_path: Path) -> Optional[Dict]:
    """
    Load YOLO data.yaml configuration.
    
    Args:
        yaml_path: Path to data.yaml file
        
    Returns:
        Dictionary with dataset configuration or None if error
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        return None


def get_image_paths_from_dataset(data_yaml_path: Path, split: str = 'train') -> List[Path]:
    """
    Get image paths from dataset configuration.
    
    Args:
        data_yaml_path: Path to data.yaml file
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        List of image paths
    """
    data = load_data_yaml(data_yaml_path)
    if not data:
        return []
    
    # Get base path from data.yaml location
    yaml_dir = data_yaml_path.parent
    
    # Check if there's a 'path' field in data.yaml (dataset root)
    if 'path' in data:
        # The dataset root path is relative to the yaml file location
        dataset_root = yaml_dir / data['path']
    else:
        # If no path field, assume dataset is in yaml directory
        dataset_root = yaml_dir
    
    # Get split path
    if split in data:
        # Split path is relative to dataset root
        split_path = dataset_root / data[split]
        
        if split_path.exists():
            # The split path should already point to the images folder
            # (e.g., train/images, val/images, test/images)
            if split_path.is_dir():
                # Get all image files
                image_files = []
                for ext in IMAGE_EXTENSIONS:
                    image_files.extend(split_path.glob(f'*{ext}'))
                    image_files.extend(split_path.glob(f'*{ext.upper()}'))
                # Remove duplicates and sort
                unique_files = sorted(set(image_files))
                return unique_files
    
    return []


def get_annotation_path(image_path: Path) -> Path:
    """
    Get corresponding annotation path for an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Path to annotation file (may not exist)
    """
    # Replace 'images' with 'labels' in path
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == 'images':
            parts[i] = 'labels'
            break
    
    # Change extension to .txt
    ann_path = Path(*parts).with_suffix('.txt')
    return ann_path


def normalize_bbox(x: float, y: float, w: float, h: float, 
                  img_width: float, img_height: float) -> Tuple[float, float, float, float]:
    """
    Convert pixel coordinates to normalized YOLO format.
    
    Args:
        x, y: Top-left corner in pixels
        w, h: Width and height in pixels
        img_width, img_height: Image dimensions
        
    Returns:
        Normalized (x_center, y_center, width, height)
    """
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    norm_width = w / img_width
    norm_height = h / img_height
    
    return x_center, y_center, norm_width, norm_height


def denormalize_bbox(x_center: float, y_center: float, width: float, height: float,
                    img_width: float, img_height: float) -> Tuple[float, float, float, float]:
    """
    Convert normalized YOLO coordinates to pixel coordinates.
    
    Args:
        x_center, y_center, width, height: Normalized YOLO format
        img_width, img_height: Image dimensions
        
    Returns:
        Pixel coordinates (x, y, w, h) where x,y is top-left corner
    """
    w = width * img_width
    h = height * img_height
    x = (x_center * img_width) - (w / 2)
    y = (y_center * img_height) - (h / 2)
    
    return x, y, w, h