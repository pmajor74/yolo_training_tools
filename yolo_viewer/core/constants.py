"""Application constants and configuration."""

from enum import Enum, auto

# Application info
APP_NAME = "Majorsoft YOLO Dataset Viewer"
APP_VERSION = "2.0.0"
ORGANIZATION = "YOLOTools"

# Window defaults
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 900
MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 600

# UI Constants
THUMBNAIL_SIZE = 150
THUMBNAIL_SPACING = 10
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.65

# File patterns
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
ANNOTATION_EXTENSION = '.txt'
MODEL_EXTENSION = '.pt'

# Import color generator for advanced color management
from ..utils.color_generator import AnnotationColorManager, ColorGenerator

# Initialize color manager with support for up to 80 classes
# Default theme is Midnight Blue
from ..utils.theme_constants import ColorTheme
COLOR_MANAGER = AnnotationColorManager(num_classes=80, theme=ColorTheme.MIDNIGHT_BLUE)

# Legacy colors (RGB tuples) - kept for backward compatibility
# These are now replaced by the colorblind-friendly palette
ANNOTATION_COLORS_LEGACY = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Rose
]

# Generate colorblind-friendly colors
ANNOTATION_COLORS = [COLOR_MANAGER.get_color(i) for i in range(80)]

# Annotation display
ANNOTATION_LINE_WIDTH = 2
ANNOTATION_FONT_SIZE = 12
BELOW_THRESHOLD_COLOR = (255, 0, 0)  # Red for below threshold
BELOW_THRESHOLD_LINE_WIDTH = 1

class AppMode(Enum):
    """Application modes."""
    DATASET_EDITOR = auto()
    FOLDER_BROWSER = auto()
    TRAINING = auto()
    AUTO_ANNOTATION = auto()
    MODEL_MANAGEMENT = auto()
    DATASET_SPLIT = auto()

# Mode display names
MODE_NAMES = {
    AppMode.DATASET_EDITOR: "Dataset Editor",
    AppMode.FOLDER_BROWSER: "Folder Browser",
    AppMode.TRAINING: "Training",
    AppMode.AUTO_ANNOTATION: "Auto-Annotation",
    AppMode.MODEL_MANAGEMENT: "Model Management",
    AppMode.DATASET_SPLIT: "Dataset Split"
}

# Training presets
TRAINING_PRESETS = {
    "Quick": {
        "epochs": 5,
        "batch_size": 8,
        "image_size": 640,
        "description": "Quick validation training"
    },
    "Standard": {
        "epochs": 25,
        "batch_size": 4,
        "image_size": 1280,
        "description": "Standard training run"
    },
    "Full": {
        "epochs": 100,
        "batch_size": 2,
        "image_size": 1280,
        "description": "Full production training"
    }
}