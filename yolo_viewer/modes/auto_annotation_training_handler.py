"""Training integration for auto-annotation mode."""

from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

from PyQt6.QtWidgets import QMessageBox, QTextEdit
from PyQt6.QtCore import QObject, pyqtSignal, QProcess

from ..utils.training_process import TrainingProcess
from ..core import ModelCache
from .auto_annotation_data_classes import WorkflowState


class TrainingHandler(QObject):
    """Handles model training operations for auto-annotation mode."""
    
    # Signals
    trainingStarted = pyqtSignal(str)  # config path
    trainingProgress = pyqtSignal(int, int, dict)  # epoch, total_epochs, metrics
    trainingCompleted = pyqtSignal(str)  # model path
    trainingError = pyqtSignal(str)  # error message
    logMessage = pyqtSignal(str)  # log output
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self._training_process: Optional[TrainingProcess] = None
        self._training_start_time: Optional[datetime] = None
        self._epoch_start_time: Optional[datetime] = None
        self._epoch_times = []  # List of epoch durations for ETA calculation
        
    @property
    def is_training(self) -> bool:
        """Check if training is currently in progress."""
        return bool(self._training_process and 
                    hasattr(self._training_process, '_process') and 
                    self._training_process._process and 
                    self._training_process._process.state() == QProcess.ProcessState.Running)
    
    def start_training(self, dataset_yaml_path: Path, workflow_state: WorkflowState,
                      training_config: Dict) -> bool:
        """
        Start training process.
        
        Args:
            dataset_yaml_path: Path to dataset YAML file
            workflow_state: Current workflow state
            training_config: Training configuration dictionary with parameters like:
                - model: Model name (e.g., 'yolov8n.pt')
                - epochs: Number of epochs
                - batch: Batch size
                - imgsz: Image size
                - lr0: Learning rate
                - device: Device to use ('cpu' or 'cuda')
                
        Returns:
            True if training started successfully, False otherwise
        """
        if not dataset_yaml_path:
            QMessageBox.warning(self._parent, "No Dataset", 
                              "Please load or create a dataset first.")
            return False
        
        if self.is_training:
            QMessageBox.warning(self._parent, "Training in Progress", 
                              "Training is already in progress.")
            return False
        
        # Create training process
        self._training_process = TrainingProcess()
        self._training_process.logMessage.connect(self._on_training_output)
        self._training_process.progressUpdate.connect(self._on_training_progress)
        self._training_process.trainingCompleted.connect(self._on_training_completed)
        self._training_process.trainingFailed.connect(self._on_training_error)
        self._training_process.metricsUpdate.connect(self._on_metrics_update)
        
        # Prepare training config with all required parameters
        config = {
            'data': str(dataset_yaml_path),
            'model': training_config.get('model', 'yolov8n.pt'),
            'epochs': training_config.get('epochs', 100),
            'batch': training_config.get('batch', 16),
            'imgsz': training_config.get('imgsz', 640),
            'lr0': training_config.get('lr0', 0.01),
            'device': training_config.get('device', 'cpu'),
            'project': 'runs/train',
            'name': f'auto_annotation_iter_{workflow_state.iteration}',
            'patience': 50,  # Early stopping patience
            'augment': False,  # Enable augmentation
            'cache': False,  # Don't cache images in memory
            'exist_ok': True  # Allow overwriting existing runs
        }
        
        # Create output directory
        output_dir = Path('runs/train') / f'auto_annotation_iter_{workflow_state.iteration}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start training
        self._training_start_time = datetime.now()
        self._epoch_times = []
        self._epoch_start_time = datetime.now()
        self._training_process.start_training(config, output_dir)
        
        self.trainingStarted.emit(str(dataset_yaml_path))
        
        return True
    
    def stop_training(self):
        """Stop training process."""
        if self._training_process:
            self._training_process.stop_training()
    
    def _on_training_output(self, output: str):
        """Handle training output."""
        self.logMessage.emit(output)
    
    def _on_metrics_update(self, metrics: str):
        """Handle metrics update from training process."""
        # Re-emit as log message so charts can parse it
        self.logMessage.emit(metrics)
    
    def _on_training_progress(self, epoch: int, total_epochs: int):
        """Handle training progress update."""
        # Calculate epoch time and ETA
        eta_str = ""
        if self._epoch_start_time and epoch > 0:
            # Record time for completed epoch
            epoch_duration = (datetime.now() - self._epoch_start_time).total_seconds()
            self._epoch_times.append(epoch_duration)
            self._epoch_start_time = datetime.now()
            
            # Calculate average epoch time
            avg_epoch_time = sum(self._epoch_times) / len(self._epoch_times)
            
            # Calculate ETA
            remaining_epochs = total_epochs - epoch
            eta_seconds = remaining_epochs * avg_epoch_time
            eta = timedelta(seconds=int(eta_seconds))
            eta_str = str(eta)
            
        metrics = {'eta': eta_str}
        self.trainingProgress.emit(epoch, total_epochs, metrics)
    
    def _on_training_completed(self, model_path: str):
        """Handle training completion."""
        self.trainingCompleted.emit(model_path)
    
    def _on_training_error(self, error: str):
        """Handle training error."""
        self.trainingError.emit(error)
    
    def show_training_error(self, error: str, console_output: str = ""):
        """
        Show detailed training error dialog.
        
        Args:
            error: Error message
            console_output: Console output for debugging
        """
        # Create a more detailed error message
        error_details = f"Training failed: {error}\n\n"
        
        # Add last few lines of console output if available
        if console_output:
            lines = console_output.strip().split('\n')
            # Get last 10 lines of output
            recent_output = '\n'.join(lines[-10:])
            error_details += "Recent console output:\n" + recent_output
        
        # Create a message box with scrollable text
        msg_box = QMessageBox(self._parent)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Training Error")
        msg_box.setText("Training failed. Click 'Show Details' for more information.")
        msg_box.setDetailedText(error_details)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        # Make the detailed text area larger
        msg_box.setStyleSheet("QTextEdit { min-width: 600px; min-height: 300px; }")
        
        msg_box.exec()
    
    def load_model_to_cache(self, model_path: str, workflow_enabled: bool = False):
        """
        Load trained model into model cache.
        
        Args:
            model_path: Path to model file
            workflow_enabled: Whether workflow automation is enabled
        """
        model_cache = ModelCache()
        model_cache.load_model(model_path)
        
        if not workflow_enabled:
            QMessageBox.information(self._parent, "Training Complete", 
                                  f"Training completed!\nModel saved to: {model_path}")