# YOLO Training Tools

A comprehensive application for training YOLO object detection models. This tool streamlines the entire workflow from dataset creation through model deployment with an intuitive visual interface.

## Why Use Yet Another Data Annotation Tool?

This particular tool was designed to completely run from your PC. In environments where you are unable to use publically available resources such as Roboform due to say enterprise restrictions, you are still able to do your labelling work relatively easy with this tool for free.

## Author
Patrick Major

## Tutorial
- [Video Tutorial Showing How to Annotate, Train and Auto-annotate for yolo models](https://youtu.be/Jt-mAiah-sc)
- [Text based tutorial](https://github.com/pmajor74/yolo_training_tools/blob/main/readme.tutorial.md)

## Python Versions
- Python 3.10 to Pyton 3.12 (as per https://docs.ultralytics.com/quickstart).
- This has been tested working on Windows with both Pytorch CPU (default with Ultralytics) and Nvidia GPU (CUDA 12.8).
- Linux testing pending, please give feedack if you have this running on Linux.

## Upcoming Features
- N8N integration examples

## Contributions
Just drop me a line at codingwithoutbugs@gmail.com and let me now how you're using this. 

# Note
While this should work on a linux based system, it has only been tested using Windows.

### GPU Support

## 🌟 Features

- **Visual Dataset Editor** - Create and modify bounding box annotations with an intuitive drawing interface
- **Folder Browser Mode** - Browse any folder and run inference or annotate images without dataset structure
- **Auto-Annotation System** - Leverage existing models to accelerate dataset expansion with confidence-based review
- **Real-time Training** - Monitor training progress with live charts, metrics, and customizable augmentation
- **Model Management** - Easy model loading and switching between different weights
- **Dataset Splitting** - Automatically organize datasets into train/val/test sets with validation
- **Dark Theme UI** - Eye-friendly interface for extended annotation sessions
- **GPU/CPU Status Indicator** - Real-time display of compute device in status bar

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Features Overview](#-features-overview)
  - [Auto-Annotation Mode](#auto-annotation-mode) - Main quick dataset annotation mode.
  - [Model Management](#model-management)
  - [Dataset Editor](#dataset-editor)
  - [Folder Browser Mode](#folder-browser-mode)
  - [Training Mode](#training-mode)  
  - [Dataset Split Mode](#dataset-split-mode)
- [Keyboard Shortcuts](#-keyboard-shortcuts)
- [Dataset Structure](#-dataset-structure)
- [Workflows](#-workflows)
  - [Complete Dataset Creation Workflow](#complete-dataset-creation-workflow)
  - [Auto-Annotation Workflow](#auto-annotation-workflow)
- [Tips and Best Practices](#-tips-and-best-practices)
- [Troubleshooting](#-troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Windows, Linux, or macOS
- CUDA-capable GPU (optional but recommended for faster training), CPU works, just slower.
   - NOTE: If you only have a CPU, or a low GPU RAM (say 8BG or lower), you may have to lower the batche size from 16 to 8 or 4.

### Quick Install (Windows)

1. Clone the repository:
```bash
git clone https://github.com/pmajor74/yolo8_training_tools.git
cd yolo8_training_tools
```

2. Run the installer:
```bash
installer.cmd
```

This will automatically create a virtual environment and install all dependencies using UV package manager. If you want to use pip, just open the installer.cmd and remove the UV from the pip install line or follow the manual installation steps.

### Manual Installation

1. Create a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### GPU Support

For CUDA GPU acceleration:
1. Uninstall CPU-only PyTorch: `pip uninstall torch torchvision`
2. Install CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org) matching your CUDA version

The application will automatically detect and display your GPU in the status bar when available.

## 🎯 Quick Start

1. Launch the application:
```bash
python Yolo_TrainerTools.py
```

2. The application opens with six tabs. Start with **Model Management** to load a model, or begin creating a dataset in **Dataset Editor**.

3. For your first project:
   - Create a new dataset in Dataset Editor
   - Annotate 50-100 images manually
   - Train an initial model
   - Use Auto-Annotation to expand your dataset
   - Retrain for improved accuracy

## 📚 Features Overview

### Model Management

The central hub for loading and managing YOLO models. Always start here when working with existing models. 
You will always want to click on the "Find Models in Project" button which will search all folders within the project for the pretrained .pt files.
The various yolov models listed are default vanilla models and will be downloaded for you.

**Key Features:**
- Automatic discovery of models in the `runs/` directory
- Load external `.pt` model files from anywhere
- View model information and current status
- Quick model switching with double-click
- Support for exporting both YOLOv8 and ONNX formats

**How to Use:**
1. **Discovered Models**: Double-click any model in the list to load it
2. **External Models**: Click "📁 Browse" to load `.pt` files from other locations
3. The loaded model becomes available across all modes
4. Current model status is shown in the top toolbar

### Dataset Editor

Create and edit bounding box annotations with a visual interface designed for efficiency, as well as create the data.yaml file.

**Key Features:**
- Draw bounding boxes directly on images
- Support for multiple classes
- Undo/redo functionality (Ctrl+Z/Y)
- Batch save operations
- Thumbnail gallery for easy navigation
- Zoom and pan controls
- Real-time annotation validation

**How to Use:**
1. **Create New Dataset**: 
   - Click "➕ Create Dataset"
   - Choose save location for `data.yaml`
   - Set image folder paths
   - Define classes

2. **Load Existing Dataset**:
   - Click "📁 Load Dataset"
   - Select the `data.yaml` file

3. **Annotate Images**:
   - Select images from the left thumbnail panel
   - Click and drag to draw bounding boxes
   - Use number keys (1, 2) to switch classes
   - Right-click boxes to delete them
   - Use Ctrl+Z/Y for undo/redo

4. **Save Work**:
   - Click "💾 Save All" to save all pending changes
   - Individual saves happen automatically when switching images

**Keyboard Shortcuts:**
- `1`, `2`: Switch between classes
- `Ctrl+Z`: Undo last action
- `Ctrl+Y`: Redo action
- `Delete`: Remove selected box
- `Mouse Wheel`: Zoom in/out
- `Middle Mouse + Drag`: Pan image

### Folder Browser Mode

Browse and work with images in any folder without needing a formal dataset structure. Perfect for quick inference or ad-hoc annotation.

**Key Features:**
- Browse any folder containing images
- Run inference on selected or all images
- Create annotations on-the-fly
- Filter images by annotation status
- Export annotations in YOLO format
- Confidence threshold adjustment
- Batch operations support

**How to Use:**
1. **Browse Images**:
   - Click "📁 Browse Folder" to select any image folder
   - Use filters to show All/Annotated/Unannotated images
   - Navigate with thumbnail gallery

2. **Run Inference**:
   - Load a model first (in Model Management)
   - Select images (or use all)
   - Click "🚀 Run Inference"
   - Adjust confidence threshold with slider
   - Review and edit detections

3. **Manual Annotation**:
   - Select an image
   - Draw bounding boxes directly
   - Save annotations individually or in batch

4. **Export Results**:
   - Click "💾 Export Annotations"
   - Choose destination folder
   - Annotations save in YOLO format

### Training Mode

Train YOLO models with comprehensive configuration options and real-time monitoring.

**Key Features:**
- Visual parameter configuration
- Real-time training charts (5 different metrics)
- Advanced augmentation settings
- Automatic ONNX export option
- Early stopping support
- GPU/CPU selection
- Training pause/resume
- Model comparison tools

**Training Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| Epochs | 100 | Number of complete passes through the dataset |
| Batch Size | 16 | Images per batch (reduce for limited GPU memory) |
| Image Size | 640 | Input image resolution (width and height) |
| Learning Rate | 0.01 | Initial learning rate for optimizer |
| Patience | 20 | Epochs to wait before early stopping |

**Augmentation Options:**
The training mode includes comprehensive augmentation settings with tooltips:
- **Geometric**: Rotation, translation, scale, shear, perspective
- **Flipping**: Horizontal and vertical flip probabilities
- **Advanced**: Mosaic, mixup augmentations
- **Color**: HSV adjustments for hue, saturation, and value

**Available Charts:**
1. **Loss Overview** - Overall training loss trend
2. **Detailed Losses** - Box, classification, and DFL components
3. **Train vs Validation** - Monitor overfitting
4. **mAP Metrics** - Mean Average Precision at different IoU thresholds
5. **Precision & Recall** - Detection quality metrics

**How to Train:**
1. Load a dataset (from Dataset Editor or File menu)
2. Configure parameters in the left panel
3. Adjust augmentation settings as needed
4. Select base model:
   - `yolov8n.pt` - Fastest, good for testing
   - `yolov8s.pt` - Balanced speed/accuracy
   - `yolov8m.pt` - Higher accuracy, slower
5. Click "▶️ Start Training"
6. Monitor progress via logs and charts
7. Model saves automatically to `runs/detect/train_*/weights/best.pt`

### Auto-Annotation Mode

The crown jewel for efficient dataset expansion - use existing models to annotate new data with intelligent confidence-based review and optional workflow automation.

**Key Features:**
- Three-tier confidence system (High/Medium/Low)
- Interactive annotation editing
- Batch operations
- Quality metrics and statistics
- Category-based filtering
- Optional workflow automation
- Support for already annotated images

**Confidence Thresholds:**
| Level | Range | Color | Action |
|-------|-------|-------|--------|
| High | > 0.80 | 🟢 Green | Auto-approved |
| Medium | 0.40-0.80 | 🟡 Yellow | Needs review |
| Low | < 0.40 | 🔴 Red | Auto-rejected |

**Auto-Annotation Workflow:**
Before you can use the Auto-Annotation mode, you need to create a lightly annotated dataset and run a small batch of training against your dataset.

The reason you will use this workflow is to create a decent dataset fast, once that is done you will use the Training tab to select a better yolo model as needed and higher epoches and any data augmentation needed to improve accuracy.

0. **Pre-Setup**:
   - Use the Dataset Management tab to create your data.yaml file with all the classes you will be using
   - Use the Folder Browser to load the folder of images that you will be annotating and annotate about 25 of each type of detection classificaiton category
   - Use the Training tab to create your first lightly trained model. Change the epoches to 25 and you can most likely leave the other settings alone as we are just doing an initial pre-training
   - Create a backup copy of your training data as this mode will add annotations as well as move images to a rejected folder should you determine they are not suitable for training set anymore. Let's refer to this new folder of your image training data \image_data

1. **Setup**:
   - Load a trained model in Model Management, hit the "Find Models in Project" button to get the latest model that you just trained. Select the model at the top of the list and then click on "Load Selected Model" (or double click on the model which loads it as well)
   - Open the Auto-Annotation tab
   - Click on "Select folder" and select the folder \image_data that you created in the Pre-Setup step. This will be the same folder that all the annotations are deposited into as .txt files in the yolo format.
   - Click on the "Load Dataset" button and load the dataset data.yaml you created in the Pre-Setup

2. **Automatic Processing**:
   - Click "Start Auto-Annotation" to begin
   - Model runs inference on all images
   - Detections are categorized by confidence
   - Progress shown in real-time

3. **Review Process**:
   - The Confidence thresholds will help sort the images into ones that are likely approvable, needs review or rejected. These filters are only here for helping you organize your images into easy to export groups. The move an image (or group of selected images) into the Approved group, hit the A key, and R for rejected.
   - The Approved and Rejected filters have no effect and are only for organizing. 
   - You Click images in gallery to review/edit/delete annotation boxes. 
   - What moves you to the next stage in the auto-annotation process is to select all the annotated images that you want to export the annotation texts into the \image_data folder (or whatever you picked for folder name) and click on the "Export Selected Annotations". 
   - Once your annotations have been exported, they will be split into your "working" folder. If there is the train/val/test folders in your working folder, they will be deleted. The images with annotations will be grouped and split properly into the train/val folders. For Auto-annotation we are skipping using "test" as we have very limited images for this training.
   - Once the training is done, the new model will be loaded and the images that have no annotations in the \image_data folder will have inferrance against them using the new model
   - Do the process again until you have all your images annotated.

### Dataset Split Mode

Organize datasets into proper train/validation/test splits with automatic validation and error handling.

**Key Features:**
- Flexible split ratios (customizable percentages)
- Missing label detection
- Orphaned file cleanup
- Reproducible random splits
- Support for both flat and structured datasets
- Automatic data.yaml generation
- Validation reporting

**Dataset Types Supported:**
1. **Flat Structure**: Images and labels in same folder (Preferred)
```
dataset/
├── image1.jpg
├── image1.txt
├── image2.jpg
└── image2.txt
```

2. **Structured**: Separate images and labels folders
```
dataset/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── labels/
    ├── image1.txt
    └── image2.txt
```

**How to Use:**
1. Select dataset structure type (Flat/Structured)
2. Browse for source folder
3. Set output directory
4. Adjust split ratios (default 70/20/10)
5. Configure number of classes and names
6. Click "Run Split"

**Validation Features:**
- Detects images without corresponding labels
- Finds orphaned label files
- Creates `rejected/` folder for problematic files
- Generates detailed split statistics
- Creates proper data.yaml configuration

## ⌨️ Keyboard Shortcuts

### Global Shortcuts
| Key | Action |
|-----|--------|
| `Ctrl+O` | Open dataset |
| `Ctrl+S` | Save (context-dependent) |
| `Ctrl+Q` | Quit application |
| `Tab` | Switch between modes |

### Dataset Editor & Annotation Canvas
| Key | Action |
|-----|--------|
| `1`, `2` | Switch to class 1 or 2 |
| `Ctrl+Z` | Undo last action |
| `Ctrl+Y` | Redo action |
| `Delete` | Delete selected box |
| `Mouse Wheel` | Zoom in/out |
| `Middle Mouse + Drag` | Pan image |
| `Escape` | Deselect box |

### Training Mode
| Key | Action |
|-----|--------|
| `Space` | Pause/Resume training |
| `Escape` | Stop training |

## 📁 Dataset Structure

### YOLO Format
Each image needs a corresponding `.txt` file with the same name:

**Annotation Format:**
```
class_id x_center y_center width height
```
- All values normalized to 0.0-1.0
- One line per object
- Space-separated values

**Example `image1.txt`:**
```
0 0.523 0.445 0.200 0.300
1 0.701 0.823 0.150 0.180
```

### data.yaml Configuration
```yaml
# Paths relative to this file
train: train/images
val: val/images
test: test/images  # optional

# Number of classes
nc: 2

# Class names
names:
  0: QR
  1: DATAMATRIX
```

## 🔄 Workflows

### Complete Dataset Creation Workflow

1. **Initial Dataset Creation** (Dataset Editor)
   - Create new dataset structure
   - Manually annotate 50-100 diverse images
   - Ensure good examples of all classes
   - Save annotations frequently

2. **First Model Training** (Training Mode)
   - Start with yolov8n for quick iteration
   - Train for 50-100 epochs
   - Enable augmentations for robustness
   - Monitor validation metrics
   - Stop when loss plateaus

3. **Dataset Expansion** (Auto-Annotation Mode)
   - Collect 200-500 new unlabeled images
   - Run auto-annotation with trained model
   - Focus review on medium-confidence detections
   - Export approved annotations

4. **Merge and Split** (Dataset Split Mode)
   - Combine original and auto-annotated data
   - Create new train/val/test splits
   - Maintain balanced class distribution

5. **Retrain with Expanded Dataset**
   - Use larger model (yolov8s or yolov8m)
   - Train for more epochs
   - Compare metrics with baseline

6. **Iterate**
   - Repeat steps 3-5 until desired accuracy
   - Each iteration improves model performance
   - Focus on edge cases and errors

### Auto-Annotation Workflow

The auto-annotation feature creates a powerful feedback loop for rapid dataset expansion:

```
Manual Annotation (50-100 images)
        ↓
Train Initial Model
        ↓
Auto-Annotate New Images (500+)
        ↓
Human Review & Correction
        ↓
Expand Training Dataset
        ↓
Retrain Improved Model
        ↓
    (Repeat)
```

**Efficient Review Strategy:**

1. **First Pass - Yellow (Medium Confidence)**:
   - These benefit most from human review
   - Often just need minor adjustments
   - May contain valid detections needing confirmation

2. **Second Pass - Green (High Confidence)**:
   - Quick scan for systematic errors
   - Check for missed objects
   - Verify class assignments

3. **Third Pass - Red (Low Confidence)**:
   - May contain difficult but valid detections
   - Good source of hard training examples
   - Often partial or occluded objects

**Quality Control Tips:**
- Monitor class distribution in statistics
- Ensure balanced dataset composition
- Track approval/rejection rates
- Document common error patterns

## 💡 Tips and Best Practices

### Dataset Management
- **Consistency**: Maintain uniform annotation style across all images
- **Diversity**: Include various lighting, angles, backgrounds, and distances
- **Balance**: Aim for roughly equal numbers of each class
- **Validation**: Keep 20% of data separate for unbiased evaluation
- **Edge Cases**: Specifically include difficult examples (blur, occlusion, etc.)

### Annotation Best Practices
- **Tight Boxes**: Minimize background, include only the barcode
- **Complete Objects**: Don't cut off parts of codes
- **Occlusion**: Annotate visible portions of partially hidden codes
- **Small Objects**: Zoom in for precise annotation
- **Regular Saves**: Save after every 10-20 annotations

### Training Optimization
- **Start Small**: Use yolov8n for rapid experimentation
- **Batch Size**: Largest that fits in GPU memory (8-32 typical)
- **Augmentation**: Enable for better generalization
- **Early Stopping**: Use patience of 20-30 epochs
- **Learning Rate**: Start with default, reduce if loss explodes

### Performance Tips
- **GPU Usage**: Always prefer GPU (10-50x faster)
- **Image Size**: 640 is good balance, 416 for speed, 1280 for accuracy
- **Cache Images**: Enable for datasets under 10GB
- **Multi-Scale**: Enable for varying object sizes

## 🔧 Troubleshooting

### Common Issues and Solutions

**"No module named 'ultralytics'"**
- Solution: Run `pip install -r requirements.txt`

**"CUDA out of memory"**
- Reduce batch size (try 8, 4, or 2)
- Reduce image size to 512 or 416
- Close other GPU applications
- Use CPU if necessary

**"No model loaded" error**
- Go to Model Management tab first
- Load a model before using other features
- Check model file path is correct

**Auto-annotation produces poor results**
- Train model for more epochs
- Increase dataset diversity
- Adjust confidence thresholds
- Add more training data

**Training crashes or stops**
- Check available disk space
- Verify dataset paths in data.yaml
- Look for corrupted images
- Check console for errors

**Device shows CPU instead of GPU**
- Verify CUDA installation
- Check PyTorch GPU support: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall CUDA-enabled PyTorch

### Performance Optimization

**Slow Training:**
1. Verify GPU is being used (check status bar)
2. Reduce image size or batch size
3. Enable image caching
4. Use SSD for dataset storage

**Slow Inference:**
1. Use GPU if available
2. Reduce image size
3. Process fewer images at once
4. Use lighter model (yolov8n)

## 📊 Understanding Metrics

### Training Metrics
- **Loss**: Lower is better, should decrease over time
- **mAP@0.5**: Mean Average Precision at 50% IoU (main metric)
- **mAP@0.5:0.95**: Stricter metric, average across IoU thresholds
- **Precision**: Ratio of correct detections to total detections
- **Recall**: Ratio of objects found to total objects

### When to Stop Training
- Validation loss stops decreasing
- mAP metrics plateau
- Large gap between train/val loss (overfitting)
- Reached target performance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the  Apache 2.0 License - see the LICENSE file for details. Created by Patrick Major. Please retain attribution in any forks or distributions.

## 🙏 Acknowledgments

- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- UI powered by [PyQt6](https://www.riverbankcomputing.com/software/pyqt/)
- Charts by [pyqtgraph](https://www.pyqtgraph.org/)
- Fast package management by [uv](https://github.com/astral-sh/uv)

---

For more information or support, please open an issue on GitHub.