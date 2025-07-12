"""Training process management utilities."""

import sys
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from PyQt6.QtCore import QObject, QProcess, pyqtSignal, pyqtSlot
from ..core import SettingsManager


class TrainingProcess(QObject):
    """Manages YOLO training process execution."""
    
    # Signals
    logMessage = pyqtSignal(str)
    progressUpdate = pyqtSignal(int, int)  # current_epoch, total_epochs
    epochProgress = pyqtSignal(int)  # percent within epoch
    batchProgress = pyqtSignal(int, int)  # current_batch, total_batches in epoch
    metricsUpdate = pyqtSignal(str)  # metrics string
    trainingCompleted = pyqtSignal(str)  # model path
    trainingFailed = pyqtSignal(str)  # error message
    stepInfoDetected = pyqtSignal(int)  # steps per epoch detected from output
    trainingStopped = pyqtSignal(str)  # message when training is stopped by user
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._current_epoch = 0
        self._total_epochs = 0
        self._output_dir: Optional[Path] = None
        self._training_start_time: Optional[datetime] = None
        self._steps_per_epoch_detected = False
        self._stopped_by_user = False
        self._current_batch_in_epoch = 0  # Track actual batch number
    
    def start_training(self, config: Dict, output_dir: Path, export_onnx: bool = False):
        """Start the training process."""
        self._output_dir = output_dir
        self._training_start_time = datetime.now()
        self._current_epoch = 0
        self._total_epochs = config.get('epochs', 100)
        self._steps_per_epoch_detected = False  # Reset detection flag
        self._stopped_by_user = False  # Reset stop flag
        
        # Create the training script
        script_path = output_dir / "train_script.py"
        self._create_training_script(script_path, config, export_onnx)
        
        # Create QProcess
        self._process = QProcess()
        self._process.setWorkingDirectory(str(Path.cwd()))
        
        # Connect signals
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.readyReadStandardError.connect(self._on_error)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error_occurred)
        
        # Start the process
        python_exe = sys.executable
        self.logMessage.emit(f"Starting training with: {python_exe} {script_path}")
        self._process.start(python_exe, [str(script_path)])
        
        if not self._process.waitForStarted(5000):
            self.logMessage.emit("ERROR: Failed to start training process")
            self._on_finished(-1, QProcess.ExitStatus.CrashExit)
    
    def stop_training(self):
        """Stop the training process."""
        if self._process and self._process.state() == QProcess.ProcessState.Running:
            self._stopped_by_user = True
            self._process.terminate()
            if not self._process.waitForFinished(5000):
                self._process.kill()
    
    def is_running(self) -> bool:
        """Check if training is running."""
        return self._process is not None and self._process.state() == QProcess.ProcessState.Running
    
    def _create_training_script(self, script_path: Path, config: Dict, export_onnx: bool):
        """Create a Python script to run YOLO training."""
        # Get workers setting before creating the script
        settings = SettingsManager()
        workers = settings.get('data_loading_workers', 4)  # Default to 4 for all platforms
        
        script_content = f'''#!/usr/bin/env python3
"""Auto-generated YOLO training script."""
import os
import sys
import traceback
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"ERROR: Failed to import ultralytics: {{e}}")
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

def main():
    # Training configuration
    config = {config}
    
    # Validate data.yaml path
    data_path = Path(config['data'])
    if not data_path.exists():
        print(f"ERROR: data.yaml not found at: {{data_path}}")
        sys.exit(1)
    
    print(f"Using data.yaml: {{data_path}}")
    
    # Validate dataset structure
    import yaml
    try:
        with open(data_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
        
        # Check for required fields
        if 'train' not in data_yaml:
            print("ERROR: 'train' path not found in data.yaml")
            sys.exit(1)
        if 'val' not in data_yaml:
            print("ERROR: 'val' path not found in data.yaml")
            sys.exit(1)
        if 'nc' not in data_yaml:
            print("ERROR: 'nc' (number of classes) not found in data.yaml")
            sys.exit(1)
        if 'names' not in data_yaml:
            print("ERROR: 'names' (class names) not found in data.yaml")
            sys.exit(1)
            
        print(f"Dataset has {{data_yaml['nc']}} classes: {{data_yaml['names']}}")
        
    except Exception as e:
        print(f"ERROR: Failed to parse data.yaml: {{e}}")
        sys.exit(1)
    
    # Initialize model
    print(f"Loading model: {{config['model']}}")
    try:
        model = YOLO(config['model'])
    except Exception as e:
        print(f"ERROR: Failed to load model '{{config['model']}}': {{e}}")
        print("\\nAvailable models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt")
        sys.exit(1)
    
    # Workers setting (passed from parent process)
    workers = {workers}
    
    # Start training
    print("Starting training...")
    print(f"Using device: {{config['device']}}")
    print(f"Using {{workers}} dataloader workers")
    
    results = model.train(
        data=config['data'],
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        lr0=config['lr0'],
        patience=config['patience'],
        augment=config['augment'],
        cache=config['cache'],
        project=config['project'],
        name=config['name'],
        exist_ok=config['exist_ok'],
        device=config['device'],
        verbose=True,
        val=True,  # Enable validation
        save_period=config.get('save_period', -1),  # Save checkpoint frequency
        workers=workers  # Set workers based on OS
    )
    
    print("Training completed!")
    print(f"Best model saved to: {{results.save_dir}}/weights/best.pt")
    
    # Export to ONNX if requested
    if {export_onnx}:
        print("Exporting to ONNX format...")
        model_path = Path(results.save_dir) / "weights" / "best.pt"
        export_model = YOLO(str(model_path))
        export_model.export(format='onnx', imgsz=config['imgsz'])
        print("ONNX export completed!")

if __name__ == '__main__':
    # Required for Windows multiprocessing
    if os.name == 'nt':
        import multiprocessing
        multiprocessing.freeze_support()
    
    try:
        main()
    except Exception as e:
        print(f"\\nERROR: Training script failed: {{type(e).__name__}}: {{e}}")
        print("\\nTraceback:")
        traceback.print_exc()
        sys.exit(1)
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix
        if os.name != 'nt':
            os.chmod(script_path, 0o755)
    
    @pyqtSlot()
    def _on_output(self):
        """Handle process output."""
        if not self._process:
            return
            
        # Read output
        output = self._process.readAllStandardOutput()
        text = output.data().decode('utf-8', errors='ignore')
        
        # Parse and emit output
        for line in text.strip().split('\n'):
            if line:
                self.logMessage.emit(line)
                self._parse_output(line)
    
    @pyqtSlot()
    def _on_error(self):
        """Handle process errors."""
        if not self._process:
            return
            
        # Read error output
        error = self._process.readAllStandardError()
        text = error.data().decode('utf-8', errors='ignore')
        
        # Process stderr output - emit all of it for debugging
        for line in text.strip().split('\n'):
            if line:
                # Always emit the line so we can see what's happening
                self.logMessage.emit(line)
                # Also parse this line for progress updates since YOLO sends progress to stderr
                self._parse_output(line)
    
    @pyqtSlot(int, QProcess.ExitStatus)
    def _on_finished(self, exit_code, exit_status):
        """Handle process completion."""
        # Final time update first
        if self._training_start_time:
            elapsed = datetime.now() - self._training_start_time
            self.logMessage.emit(f"\nTotal training time: {str(elapsed).split('.')[0]}")
        
        if self._stopped_by_user:
            # User stopped the training - this is not an error
            self.logMessage.emit("\n" + "=" * 60)
            self.logMessage.emit("TRAINING STOPPED BY USER")
            # Check if we have a partial model
            model_path = self._output_dir / "weights" / "best.pt"
            message = "Training was stopped by user.\n\n"
            if model_path.exists():
                self.logMessage.emit(f"Partial model saved to: {model_path}")
                message += f"Partial model saved to:\n{model_path}\n\n"
            
            # Add epoch information
            if self._current_epoch > 0:
                message += f"Completed {self._current_epoch} epochs out of {self._total_epochs} total epochs."
            else:
                message += "Training was stopped before completing the first epoch."
            
            self.logMessage.emit("=" * 60)
            # Emit the trainingStopped signal
            self.trainingStopped.emit(message)
            return
        elif exit_code == 0:
            # Training completed successfully
            model_path = self._output_dir / "weights" / "best.pt"
            if model_path.exists():
                self.logMessage.emit("\n" + "=" * 60)
                self.logMessage.emit("TRAINING COMPLETED SUCCESSFULLY!")
                self.logMessage.emit(f"Best model saved to: {model_path}")
                self.logMessage.emit("=" * 60)
                self.trainingCompleted.emit(str(model_path))
            else:
                self.logMessage.emit("\nWARNING: Training finished but model file not found")
                self.trainingFailed.emit("Model file not found")
        else:
            # Training failed
            self.logMessage.emit(f"\nTraining failed with exit code: {exit_code}")
            
            # Try to provide more specific error information
            error_msg = f"Exit code: {exit_code}"
            if exit_code == 1:
                error_msg += "\n\nCommon causes:\n"
                error_msg += "- Invalid data.yaml path or format\n"
                error_msg += "- Missing training images or labels\n"
                error_msg += "- Incorrect model name\n"
                error_msg += "- Python environment issues\n"
                error_msg += "\nCheck the console output for details."
            
            self.trainingFailed.emit(error_msg)
    
    @pyqtSlot(QProcess.ProcessError)
    def _on_error_occurred(self, error):
        """Handle process errors."""
        # If stopped by user, don't report as error
        if self._stopped_by_user and error == QProcess.ProcessError.Crashed:
            return
            
        error_messages = {
            QProcess.ProcessError.FailedToStart: "Failed to start training process",
            QProcess.ProcessError.Crashed: "Training process crashed",
            QProcess.ProcessError.Timedout: "Training process timed out",
            QProcess.ProcessError.WriteError: "Error writing to training process",
            QProcess.ProcessError.ReadError: "Error reading from training process",
            QProcess.ProcessError.UnknownError: "Unknown error occurred"
        }
        
        self.logMessage.emit(f"\nPROCESS ERROR: {error_messages.get(error, 'Unknown error')}")
        if not self._stopped_by_user:
            self.trainingFailed.emit(error_messages.get(error, 'Unknown error'))
    
    def _parse_output(self, line: str):
        """Parse training output for progress updates."""
        # Remove ANSI escape codes if present
        import re
        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        
        # Debug: Log lines that contain training progress indicators - commented out to reduce noise
        # if "640:" in line or ("/" in line and any(x in line for x in ["epoch", "%", "|"])):
        #     self.logMessage.emit(f"[DEBUG] Raw line: {repr(line[:100])}")
        #     self.logMessage.emit(f"[DEBUG] Clean line: {repr(clean_line[:100])}")
        
        # Parse training output - actual format from screenshot:
        # "1/5      OG      1.346      3.366      1.467      19      640:  14%|█    | 23/160 [00:15<01:35,"
        # The batch counter is AFTER the percentage and progress bar
        
        # Look for lines with "640:" which indicates training progress
        if "640:" in clean_line and "/" in clean_line:
            # Debug log to see what we're working with - commented out to reduce noise
            # self.logMessage.emit(f"[DEBUG] Found training line: {clean_line[:80]}...")
            
            # Try to find epoch at the start
            parts = clean_line.strip().split()
            epoch_found = False
            current_epoch = self._current_epoch
            total_epochs = self._total_epochs
            
            # Check first part for epoch
            if len(parts) > 0 and "/" in parts[0] and parts[0].count("/") == 1:
                try:
                    e_parts = parts[0].split("/")
                    if e_parts[0].isdigit() and e_parts[1].isdigit():
                        new_epoch = int(e_parts[0])
                        new_total = int(e_parts[1])
                        if 1 <= new_epoch <= new_total <= 1000:
                            epoch_found = True
                            if new_epoch != self._current_epoch:
                                self._current_batch_in_epoch = 0
                                current_epoch = new_epoch
                                total_epochs = new_total
                                self._current_epoch = new_epoch
                                self._total_epochs = new_total
                                self.progressUpdate.emit(self._current_epoch, self._total_epochs)
                                # Debug log - commented out
                                # self.logMessage.emit(f"[DEBUG] Emitting progressUpdate: epoch {self._current_epoch}/{self._total_epochs}")
                except Exception as e:
                    self.logMessage.emit(f"[DEBUG] Error parsing epoch: {e}")
            
            # Now look for batch counter after "640:" and "|"
            # Find the position of "640:" in the line
            idx_640 = clean_line.find("640:")
            if idx_640 >= 0:
                # Look for pattern like "14%|█    | 23/160"
                after_640 = clean_line[idx_640:]
                
                # Use regex to find the batch counter after the progress bar
                import re
                # Look for pattern: number/number after a "|"
                batch_pattern = r'\|\s*(\d+)/(\d+)'
                match = re.search(batch_pattern, after_640)
                
                if match:
                    try:
                        current_batch = int(match.group(1))
                        total_batches = int(match.group(2))
                        
                        # self.logMessage.emit(f"[DEBUG] Found batch info: {current_batch}/{total_batches}")
                        
                        # Validate batch numbers (lowered minimum from 10 to 1 to support small datasets)
                        if 1 <= current_batch <= total_batches and total_batches >= 1:
                            # Update batch progress
                            if current_batch != self._current_batch_in_epoch:
                                self._current_batch_in_epoch = current_batch
                                self.batchProgress.emit(current_batch, total_batches)
                                # Debug log for first few batches - commented out
                                # if current_batch <= 3:
                                #     self.logMessage.emit(f"[DEBUG] Emitting batchProgress: {current_batch}/{total_batches}")
                                
                                # Log key updates - removed to reduce noise
                            
                            # Calculate and emit percentage
                            percent = int((current_batch / total_batches) * 100)
                            self.epochProgress.emit(percent)
                            
                            # Detect steps per epoch
                            if not self._steps_per_epoch_detected:
                                self._steps_per_epoch_detected = True
                                self.stepInfoDetected.emit(total_batches)
                    except Exception as e:
                        self.logMessage.emit(f"[DEBUG] Error parsing batch: {e}")
        
        
        # Parse metrics from training output
        # Training loss pattern: "640: 64%" followed by losses
        if "640:" in clean_line and "%" in clean_line:
            try:
                metrics = []
                parts = clean_line.split()
                
                # Try to find loss values (usually appear as floating point numbers)
                loss_values = []
                for i, part in enumerate(parts):
                    try:
                        # Skip parts that contain units or special characters
                        if any(x in part for x in ['it/s', 's/it', '%', '|', '[', ']', ':', 'ms', 'MB']):
                            continue
                        # Only process strings that look like valid numbers
                        # Must contain only digits, dots, and optionally e/E for scientific notation
                        if not re.match(r'^[\d.eE+-]+$', part):
                            continue
                        # Must contain a decimal point to be a loss value
                        if '.' not in part:
                            continue
                        # Check if this looks like a loss value (float between 0 and 10)
                        val = float(part)
                        if 0 < val < 10:  # Likely a loss value
                            loss_values.append(val)
                    except ValueError:
                        continue
                
                # If we found loss values, format them
                if loss_values:
                    # Typically: total_loss, box_loss, cls_loss, dfl_loss
                    if len(loss_values) >= 1:
                        metrics.append(f"Loss: {loss_values[0]:.3f}")
                    if len(loss_values) >= 3:
                        metrics.append(f"Box: {loss_values[1]:.3f}")
                        metrics.append(f"Cls: {loss_values[2]:.3f}")
                
                # Check for learning rate in the line (often appears with losses)
                # Format: "lr0: 0.00123" or similar
                lr_match = re.search(r'lr\d?:\s*([\d.e-]+)', clean_line, re.IGNORECASE)
                if lr_match:
                    metrics.append(f"lr: {float(lr_match.group(1)):.6f}")
                
                if metrics:
                    self.metricsUpdate.emit(" | ".join(metrics[:3]))
            except (ValueError, IndexError):
                pass
        
        # Parse validation metrics - these appear during validation phase
        # Format: "all    123    456    0.789    0.654    0.543    0.321"
        # Columns: Class, Images, Instances, P, R, mAP50, mAP50-95
        if "all" in clean_line and len(clean_line.split()) >= 7:
            parts = clean_line.split()
            if parts[0] == "all":
                try:
                    # Try to parse the metrics
                    images = int(parts[1])
                    instances = int(parts[2])
                    precision = float(parts[3])
                    recall = float(parts[4])
                    map50 = float(parts[5])
                    map50_95 = float(parts[6])
                    
                    # Emit validation metrics
                    val_metrics = []
                    val_metrics.append(f"P: {precision:.3f}")
                    val_metrics.append(f"R: {recall:.3f}")
                    val_metrics.append(f"mAP50: {map50:.3f}")
                    val_metrics.append(f"mAP: {map50_95:.3f}")
                    
                    self.metricsUpdate.emit(" | ".join(val_metrics[:3]))
                    
                    # Log validation results
                    self.logMessage.emit(f"[Validation] Precision: {precision:.3f}, Recall: {recall:.3f}, mAP@50: {map50:.3f}, mAP@50-95: {map50_95:.3f}")
                    
                    # Also emit validation epoch marker to help chart know when validation happened
                    self.metricsUpdate.emit(f"[VAL_EPOCH] Epoch {self._current_epoch}")
                except (ValueError, IndexError):
                    pass
        
        # Look for epoch validation results with losses
        # YOLO v8 format: "Epoch X/Y completed" followed by validation losses
        if "epoch" in clean_line.lower() and ("completed" in clean_line.lower() or "validation" in clean_line.lower()):
            # self.logMessage.emit(f"[VAL DEBUG] Found epoch completion marker: {clean_line[:100]}")
            # Mark that we're in validation phase
            self._in_validation = True
            
        # During validation, look for loss values
        if hasattr(self, '_in_validation') and self._in_validation:
            # Look for patterns like "val/box_loss: 0.123" or just loss values after validation header
            loss_patterns = [
                r'val/box_loss:\s*([\d.]+)',
                r'val/cls_loss:\s*([\d.]+)', 
                r'val/dfl_loss:\s*([\d.]+)',
                r'val_loss:\s*([\d.]+)',
            ]
            
            for pattern in loss_patterns:
                match = re.search(pattern, clean_line)
                if match:
                    val = float(match.group(1))
                    self.metricsUpdate.emit(f"Val Loss: {val:.3f}")
                    self._in_validation = False  # Reset after finding loss
        
        # YOLO v8 specific: Look for epoch summary with validation metrics
        # Format: "1/100      2.456      1.234      0.789      1.456      0.543      0.321      0.123"
        # This appears at the end of each epoch with train and val losses
        if self._current_epoch > 0 and "/" in clean_line and len(clean_line.split()) >= 7:
            parts = clean_line.split()
            # Check if first part is epoch indicator (e.g., "1/100")
            if len(parts) > 0 and "/" in parts[0]:
                try:
                    epoch_parts = parts[0].split("/")
                    if epoch_parts[0].isdigit() and epoch_parts[1].isdigit():
                        # This looks like an epoch summary line
                        # Try to extract validation losses (usually the last 3-4 values)
                        float_values = []
                        for part in parts[1:]:
                            try:
                                # Skip parts that contain units or special characters
                                if any(x in part for x in ['it/s', 's/it', '%', '|', '[', ']', ':', 'ms', 'MB']):
                                    continue
                                if '.' in part:
                                    # Clean the part - remove any trailing non-numeric characters
                                    clean_part = ''.join(c for c in part if c.isdigit() or c == '.')
                                    if clean_part and '.' in clean_part:
                                        val = float(clean_part)
                                        if 0 < val < 100:  # Reasonable range
                                            float_values.append(val)
                            except ValueError:
                                continue
                        
                        # If we have enough values, the last ones are typically validation losses
                        if len(float_values) >= 6:
                            # Typically: train_box, train_cls, train_dfl, val_box, val_cls, val_dfl
                            val_losses = float_values[-3:]  # Last 3 are validation losses
                            total_val_loss = sum(val_losses)
                            # self.logMessage.emit(f"[VAL DEBUG] Epoch summary detected - val losses: {val_losses}, total: {total_val_loss:.3f}")
                            self.metricsUpdate.emit(f"Val Loss: {total_val_loss:.3f}")
                except (ValueError, IndexError):
                    pass
        
        # Check for validation phase - multiple indicators
        # 1. Explicit "Validating" message
        if "validating" in clean_line.lower():
            # self.logMessage.emit(f"[VAL DEBUG] Found 'Validating' in line: {clean_line[:100]}")
            # Look for loss values on the same line or mark validation phase
            loss_match = re.search(r'loss:\s*([\d.]+)', clean_line, re.IGNORECASE)
            if loss_match:
                val_loss = float(loss_match.group(1))
                # self.logMessage.emit(f"[VAL DEBUG] Emitting Val Loss: {val_loss:.3f}")
                self.metricsUpdate.emit(f"Val Loss: {val_loss:.3f}")
            else:
                # Mark that we're in validation phase for subsequent lines
                self._in_validation_phase = True
                # self.logMessage.emit(f"[VAL DEBUG] Marked validation phase start")
                
        # 2. Lines with "val" and loss values
        elif "val" in clean_line.lower() and any(x in clean_line.lower() for x in ["loss", "box", "cls", "dfl"]):
            # self.logMessage.emit(f"[VAL DEBUG] Found 'val' + loss keywords in: {clean_line[:100]}")
            # Try to find numeric values that could be losses
            loss_values = []
            # Use regex to find all float numbers, excluding those with units
            # This pattern finds floats not followed by units
            float_pattern = r'\b(\d+\.\d+)\b(?![a-zA-Z/\[\]])'  
            matches = re.findall(float_pattern, clean_line)
            
            for match in matches:
                try:
                    val = float(match)
                    if 0 < val < 10:  # Reasonable loss range
                        loss_values.append(val)
                except ValueError:
                    continue
                    
            if loss_values:
                # self.logMessage.emit(f"[VAL DEBUG] Found loss values: {loss_values}")
                # If we have multiple values, sum them for total loss
                if len(loss_values) >= 3:
                    total_loss = sum(loss_values[:3])  # Sum box, cls, dfl
                    # self.logMessage.emit(f"[VAL DEBUG] Emitting total Val Loss: {total_loss:.3f}")
                    self.metricsUpdate.emit(f"Val Loss: {total_loss:.3f}")
                else:
                    # Just use the first value
                    # self.logMessage.emit(f"[VAL DEBUG] Emitting Val Loss: {loss_values[0]:.3f}")
                    self.metricsUpdate.emit(f"Val Loss: {loss_values[0]:.3f}")
                    
        # 3. Check for explicit validation loss patterns
        val_patterns = [
            r'val/box_loss:\s*([\d.]+)',
            r'val/cls_loss:\s*([\d.]+)',
            r'val/dfl_loss:\s*([\d.]+)',
            r'val_loss:\s*([\d.]+)',
            r'validation.*loss:\s*([\d.]+)'
        ]
        
        for pattern in val_patterns:
            match = re.search(pattern, clean_line, re.IGNORECASE)
            if match:
                val_loss = float(match.group(1))
                # self.logMessage.emit(f"[VAL DEBUG] Pattern '{pattern}' matched with value: {val_loss:.3f}")
                self.metricsUpdate.emit(f"Val Loss: {val_loss:.3f}")
                break
        
        # Also check for explicit metric labels in output
        if any(x in clean_line for x in ["mAP50:", "mAP50-95:", "Precision:", "Recall:"]):
            try:
                metrics = []
                
                # Parse individual metrics
                if "Precision:" in clean_line:
                    match = re.search(r'Precision:\s*([\d.]+)', clean_line)
                    if match:
                        metrics.append(f"P: {float(match.group(1)):.3f}")
                
                if "Recall:" in clean_line:
                    match = re.search(r'Recall:\s*([\d.]+)', clean_line)
                    if match:
                        metrics.append(f"R: {float(match.group(1)):.3f}")
                
                if "mAP50:" in clean_line and "mAP50-95:" not in clean_line:
                    match = re.search(r'mAP50:\s*([\d.]+)', clean_line)
                    if match:
                        metrics.append(f"mAP50: {float(match.group(1)):.3f}")
                
                if "mAP50-95:" in clean_line:
                    match = re.search(r'mAP50-95:\s*([\d.]+)', clean_line)
                    if match:
                        metrics.append(f"mAP: {float(match.group(1)):.3f}")
                
                if metrics:
                    self.metricsUpdate.emit(" | ".join(metrics))
            except (ValueError, IndexError):
                pass