"""Process handler for hyperparameter tuning."""

import sys
import os
import json
import yaml
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple

from PyQt6.QtCore import QObject, QProcess, pyqtSignal, pyqtSlot, QTimer


class TuningProcess(QObject):
    """Manages the hyperparameter tuning process."""
    
    # Signals
    logMessage = pyqtSignal(str)
    progressUpdate = pyqtSignal(int, int)  # current_iteration, total_iterations
    iterationComplete = pyqtSignal(int, float, dict)  # iteration, fitness, params
    tuningComplete = pyqtSignal(str)  # results_path
    errorOccurred = pyqtSignal(str)
    bestFitnessUpdate = pyqtSignal(float, dict)  # fitness, params
    timeUpdate = pyqtSignal(str, str)  # elapsed_time, eta
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._output_dir: Optional[Path] = None
        self._config: Dict = {}
        self._current_iteration = 0
        self._total_iterations = 0
        self._best_fitness = 0.0
        self._best_params = {}
        self._start_time = None
        self._iteration_times = []  # Track time per iteration for ETA calculation
        self._user_stopped = False  # Track if user initiated stop
        self._completion_timer = None  # Timer to check for stuck process after completion
        self._resume_offset = 0  # Offset for resumed sessions
        
    def start_tuning(self, config: Dict, output_dir: Path, resume: bool = False) -> bool:
        """
        Start the hyperparameter tuning process.
        
        Args:
            config: Tuning configuration dictionary
            output_dir: Directory to save results
            resume: Whether to resume from existing results
            
        Returns:
            bool: True if process started successfully
        """
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self.errorOccurred.emit("Tuning process is already running")
            return False
            
        self._config = config
        self._output_dir = output_dir
        self._current_iteration = 0
        self._total_iterations = config.get('iterations', 300)
        self._best_fitness = 0.0
        self._best_params = {}
        self._start_time = datetime.now()
        self._user_stopped = False  # Reset flag for new session
        
        # Check for existing results if resuming
        self._resume_offset = 0
        if resume:
            existing_csv = output_dir / "tune" / "tune_results.csv"
            if existing_csv.exists():
                try:
                    with open(existing_csv, 'r') as f:
                        lines = f.readlines()
                        self._resume_offset = len(lines) - 1  # Subtract header
                        self._current_iteration = self._resume_offset
                        print(f"[INFO] Resuming from iteration {self._current_iteration}")
                        self.logMessage.emit(f"Resuming from iteration {self._current_iteration}")
                except:
                    pass
        
        # Create output directory
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tuning script
        script_path = self._output_dir / "tune_script.py"
        self._create_tuning_script(script_path, config, resume)
        
        # Setup process
        self._process = QProcess()
        self._process.readyReadStandardOutput.connect(self._handle_stdout)
        self._process.readyReadStandardError.connect(self._handle_stderr)
        self._process.finished.connect(self._handle_finished)
        self._process.errorOccurred.connect(self._handle_error)
        
        # Get Python executable
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            # We're in a virtual environment
            python_exe = sys.executable
        else:
            python_exe = 'python'
            
        # Start the process
        self.logMessage.emit(f"Starting tuning with {python_exe}")
        self.logMessage.emit(f"Script: {script_path}")
        self.logMessage.emit(f"Working directory: {self._output_dir}")
        
        self._process.setWorkingDirectory(str(self._output_dir))
        self._process.start(python_exe, [str(script_path)])
        
        if not self._process.waitForStarted(5000):  # 5 second timeout
            self.errorOccurred.emit("Failed to start tuning process")
            return False
            
        self.logMessage.emit(f"Tuning process started (PID: {self._process.processId()})")
        return True
        
    def stop_tuning(self):
        """Stop the tuning process gracefully."""
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._user_stopped = True  # Mark as user-initiated stop
            self.logMessage.emit("Stopping tuning and saving results...")
            
            # Try graceful termination first (allows Ultralytics to save)
            self._process.terminate()
            
            # Give it more time to save results
            if not self._process.waitForFinished(10000):  # 10 seconds
                self.logMessage.emit("Force stopping process...")
                self._process.kill()
                self._process.waitForFinished(2000)
                
    def _create_tuning_script(self, script_path: Path, config: Dict, resume: bool = False):
        """Create the Python script for tuning."""
        # Convert config to a proper Python representation
        config_str = repr(config)
        resume_str = repr(resume)
        
        script_content = f'''
"""Auto-generated hyperparameter tuning script."""

import os
import sys
import json
import yaml
import csv
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# Configuration
config = {config_str}
resume = {resume_str}

# Fix dataset path resolution by updating Ultralytics settings temporarily
# Save current directory
original_dir = os.getcwd()
dataset_path = Path(config['data'])

# Update Ultralytics settings to use correct dataset directory
if dataset_path.exists():
    # Get the parent directory of the data.yaml file
    dataset_parent = dataset_path.parent
    print(f"[TUNING] Dataset parent directory: {{dataset_parent}}")
    
    # Load the data.yaml to check the path field
    with open(dataset_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # If there's a 'path' field, resolve it relative to the data.yaml location
    if 'path' in data_config and data_config['path']:
        base_path = Path(data_config['path'])
        if not base_path.is_absolute():
            # Make it absolute relative to the data.yaml directory
            abs_base_path = (dataset_parent / base_path).resolve()
            print(f"[TUNING] Resolved dataset root: {{abs_base_path}}")
            
            # Create a temporary data.yaml with absolute paths
            temp_data = data_config.copy()
            temp_data['path'] = str(abs_base_path)
            
            # Write temporary config
            temp_yaml_path = Path(original_dir) / 'temp_data.yaml'
            with open(temp_yaml_path, 'w') as f:
                yaml.dump(temp_data, f)
            
            config['data'] = str(temp_yaml_path)
            print(f"[TUNING] Using temporary data config: {{temp_yaml_path}}")

# Check for resume
completed_iterations = 0
if resume:
    results_csv = Path.cwd() / 'tune' / 'tune_results.csv'
    if results_csv.exists():
        try:
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                completed_iterations = len(rows)
                print(f"[TUNING] Found existing results with {{completed_iterations}} iterations completed")
                
                # Load best fitness from existing results
                if rows:
                    best_fitness = max(float(row.get('fitness', 0)) for row in rows)
                    print(f"[TUNING] Previous best fitness: {{best_fitness:.4f}}")
        except Exception as e:
            print(f"[WARNING] Could not read existing results: {{e}}")
            completed_iterations = 0

# Adjust iterations for resume
remaining_iterations = config['iterations'] - completed_iterations
if remaining_iterations <= 0:
    print(f"[TUNING] Already completed {{config['iterations']}} iterations. Nothing to do.")
    sys.exit(0)

print("[TUNING] Starting hyperparameter tuning...")
print(f"[TUNING] Model: {{config['model']}}")
print(f"[TUNING] Dataset: {{config['data']}}")
if resume and completed_iterations > 0:
    print(f"[TUNING] Resuming from iteration {{completed_iterations + 1}}/{{config['iterations']}}")
    print(f"[TUNING] Remaining iterations: {{remaining_iterations}}")
else:
    print(f"[TUNING] Iterations: {{config['iterations']}}")
print(f"[TUNING] Epochs per iteration: {{config['epochs']}}")
print(f"[TUNING] Batch size: {{config.get('batch', 16)}}")
print(f"[TUNING] Image size: {{config.get('imgsz', 640)}}")
print(f"[TUNING] Optimizer: {{config.get('optimizer', 'AdamW')}}")

# Initialize model
try:
    model = YOLO(config['model'])
    print("[TUNING] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {{e}}")
    sys.exit(1)

# Prepare search space
search_space = config.get('space', {{}})
if search_space:
    print(f"[TUNING] Search space: {{list(search_space.keys())}}")
else:
    print("[TUNING] Using default search space")

# Custom callback for progress tracking
def on_train_epoch_end(trainer):
    """Called at the end of each training epoch."""
    epoch = trainer.epoch
    if epoch % 5 == 0:  # Report every 5 epochs
        metrics = trainer.metrics
        print(f"[EPOCH] {{epoch}}/{{trainer.epochs}} - "
              f"box_loss: {{metrics.get('train/box_loss', 0):.4f}}, "
              f"cls_loss: {{metrics.get('train/cls_loss', 0):.4f}}")

def on_tune_iteration_end(tuner):
    """Called at the end of each tuning iteration."""
    iteration = len(tuner.tune_results)
    fitness = tuner.tune_results[-1].get('fitness', 0) if tuner.tune_results else 0
    params = tuner.tune_results[-1] if tuner.tune_results else {{}}
    
    print(f"[ITERATION] {{iteration}}/{{config['iterations']}} - Fitness: {{fitness:.4f}}")
    print(f"[PARAMS] {{json.dumps(params, indent=2)}}")

# Start tuning
try:
    # Use remaining iterations if resuming
    tune_iterations = remaining_iterations if resume else config.get('iterations', 300)
    
    results = model.tune(
        data=config['data'],
        epochs=config.get('epochs', 30),
        iterations=tune_iterations,
        batch=config.get('batch', 16),
        imgsz=config.get('imgsz', 640),
        optimizer=config.get('optimizer', 'AdamW'),
        space=search_space if search_space else None,
        plots=config.get('plots', True),
        save=config.get('save', True),
        val=config.get('val', True),
        project=str(Path.cwd()),
        name='tune',
        exist_ok=True,
        device=config.get('device', None),
        resume=resume  # Pass resume flag if supported by Ultralytics
    )
    
    print("[TUNING] Tuning completed successfully!")
    
    # Save results
    results_path = Path.cwd() / 'tune'
    if results_path.exists():
        print(f"[RESULTS] Results saved to: {{results_path}}")
        
        # Try to find and report best parameters
        best_params_file = results_path / 'best_hyperparameters.yaml'
        if best_params_file.exists():
            with open(best_params_file, 'r') as f:
                best_params = yaml.safe_load(f)
            print(f"[BEST] Best parameters found:")
            print(f"[BEST] {{json.dumps(best_params, indent=2)}}")
    
except KeyboardInterrupt:
    print("[TUNING] Tuning interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"[ERROR] Tuning failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Clean up temporary data.yaml if created
    if 'temp_yaml_path' in locals() and Path(temp_yaml_path).exists():
        try:
            os.remove(temp_yaml_path)
            print(f"[TUNING] Cleaned up temporary config: {{temp_yaml_path}}")
        except:
            pass

print("[TUNING] Script completed")
'''
        
        # Write script to file
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        self.logMessage.emit(f"Created tuning script: {script_path}")
        
    @pyqtSlot()
    def _handle_stdout(self):
        """Handle standard output from the process."""
        if not self._process:
            return
            
        output = self._process.readAllStandardOutput()
        text = output.data().decode('utf-8', errors='ignore')
        
        for line in text.strip().split('\n'):
            if not line:
                continue
                
            # Echo to console for debugging (per CLAUDE.md requirements)
            print(f"[TUNING] {line}")
            
            self.logMessage.emit(line)
            
            # Parse progress information
            if '[ITERATION]' in line:
                # Extract iteration number and fitness
                match = re.search(r'(\d+)/(\d+).*Fitness:\s*([\d.]+)', line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    fitness = float(match.group(3))
                    
                    # Note: [ITERATION] messages already include total iterations
                    self._current_iteration = current
                    self.progressUpdate.emit(current, total)
                    
                    if fitness > self._best_fitness:
                        self._best_fitness = fitness
                        self.bestFitnessUpdate.emit(fitness, {})
                        
            # Parse Ultralytics Tuner messages
            elif 'Tuner:' in line:
                # Parse "Tuner: X/Y iterations complete (time)"
                match = re.search(r'Tuner:\s*(\d+)/(\d+)\s+iterations complete', line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    # Adjust for resume offset
                    if hasattr(self, '_resume_offset'):
                        current += self._resume_offset
                        total = self._total_iterations  # Use original total
                    self._current_iteration = current
                    self.progressUpdate.emit(current, total)
                    
                    # Check if tuning is complete
                    if current >= total:
                        print(f"[INFO] Tuning complete: {current}/{total} iterations done")
                        self.logMessage.emit(f"Tuning complete: {current}/{total} iterations done")
                        # Give it a moment to save results then stop the process
                        QTimer.singleShot(5000, self._auto_stop_on_completion)
                    
                    # Calculate time metrics
                    if self._start_time:
                        elapsed = datetime.now() - self._start_time
                        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
                        
                        # Calculate ETA
                        if current > 0:
                            avg_time_per_iter = elapsed.total_seconds() / current
                            remaining_iters = total - current
                            eta_seconds = avg_time_per_iter * remaining_iters
                            eta = datetime.now() + timedelta(seconds=eta_seconds)
                            eta_str = eta.strftime("%H:%M:%S")
                            
                            self.timeUpdate.emit(elapsed_str, eta_str)
                    
                # Parse "Tuner: Best fitness=X.XXX observed at iteration Y"
                match = re.search(r'Best fitness=([\d.]+)\s+observed at iteration\s+(\d+)', line)
                if match:
                    fitness = float(match.group(1))
                    iteration = int(match.group(2))
                    if fitness > self._best_fitness:
                        self._best_fitness = fitness
                        self.bestFitnessUpdate.emit(fitness, {})
                        
                # Parse "Tuner: Starting iteration X/Y with hyperparameters"
                match = re.search(r'Starting iteration\s+(\d+)/(\d+)', line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    # Adjust for resume offset
                    if hasattr(self, '_resume_offset'):
                        current += self._resume_offset
                        total = self._total_iterations  # Use original total
                    self.progressUpdate.emit(current, total)
                    
                    # Check if we're at the last iteration starting
                    if current == total:
                        print(f"[DEBUG] Starting final iteration {current}/{total}")
                        # Set a timer to check for completion after a reasonable time
                        if not self._completion_timer:
                            self._completion_timer = QTimer()
                            self._completion_timer.setSingleShot(True)
                            self._completion_timer.timeout.connect(self._check_completion_timeout)
                            # Give 5 minutes for the final iteration to complete
                            self._completion_timer.start(300000)  # 5 minutes
                        
            elif '[PARAMS]' in line:
                # Try to extract parameters JSON
                try:
                    # Get the rest of the text after [PARAMS]
                    idx = text.index('[PARAMS]')
                    params_text = text[idx+8:].strip()
                    if params_text.startswith('{'):
                        # Find the end of the JSON object
                        brace_count = 0
                        end_idx = 0
                        for i, char in enumerate(params_text):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        if end_idx > 0:
                            params = json.loads(params_text[:end_idx])
                            self.iterationComplete.emit(self._current_iteration, self._best_fitness, params)
                except (json.JSONDecodeError, ValueError) as e:
                    pass  # Ignore parsing errors
                    
            elif '[BEST]' in line:
                # Best parameters found
                self.logMessage.emit(f"Best parameters: {line}")
                
            elif '[RESULTS]' in line:
                # Extract results path
                match = re.search(r'Results saved to:\s*(.+)', line)
                if match:
                    results_path = match.group(1).strip()
                    self.tuningComplete.emit(results_path)
                    
            # Check for completion messages
            elif 'Hyperparameter tuning complete' in line or 'Tuning completed successfully' in line:
                print("[INFO] Detected tuning completion message")
                self.logMessage.emit("Tuning completed successfully!")
                # Give time to save results then stop
                QTimer.singleShot(3000, self._auto_stop_on_completion)
                    
            elif '[ERROR]' in line:
                self.errorOccurred.emit(line)
                
    @pyqtSlot()
    def _handle_stderr(self):
        """Handle standard error from the process."""
        if not self._process:
            return
            
        error = self._process.readAllStandardError()
        text = error.data().decode('utf-8', errors='ignore')
        
        # Filter out progress bars and non-critical warnings
        for line in text.strip().split('\n'):
            if not line:
                continue
                
            # Skip progress bar characters
            if any(char in line for char in ['█', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '━', '│']):
                continue
                
            # Skip common non-critical warnings
            if any(skip in line.lower() for skip in ['warning', 'info:', 'debug:']):
                if 'error' not in line.lower():
                    continue
            
            # Echo stderr to console (per CLAUDE.md requirements)        
            print(f"[TUNING-STDERR] {line}")
            
            self.logMessage.emit(f"[STDERR] {line}")
            
    def _check_and_load_results(self):
        """Check for and load any saved results."""
        if not self._output_dir:
            return False
            
        results_dir = self._output_dir / "tune"
        if not results_dir.exists():
            return False
            
        # Check for key result files
        best_params_file = results_dir / "best_hyperparameters.yaml"
        results_csv = results_dir / "tune_results.csv"
        
        if best_params_file.exists() or results_csv.exists():
            # Results exist - load them
            self.logMessage.emit(f"Results found at: {results_dir}")
            
            # Count iterations completed if CSV exists
            if results_csv.exists():
                try:
                    with open(results_csv, 'r') as f:
                        lines = f.readlines()
                        iterations_done = len(lines) - 1  # Subtract header
                        self.logMessage.emit(f"Completed {iterations_done} iterations before stopping")
                except Exception as e:
                    self.logMessage.emit(f"Error reading results: {e}")
                    
            # Emit completion with results path
            self.tuningComplete.emit(str(results_dir))
            return True
        return False
    
    @pyqtSlot(int, QProcess.ExitStatus)
    def _handle_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """Handle process completion."""
        if exit_status == QProcess.ExitStatus.NormalExit:
            if exit_code == 0:
                self.logMessage.emit("Tuning completed successfully")
                self._check_and_load_results()
            else:
                if self._user_stopped:
                    self.logMessage.emit("Tuning stopped by user")
                    # Try to load any partial results
                    if self._check_and_load_results():
                        self.logMessage.emit("Partial results saved successfully")
                    else:
                        self.logMessage.emit("No results were saved")
                else:
                    self.errorOccurred.emit(f"Tuning failed with exit code {exit_code}")
        else:
            # Process was terminated or crashed
            if self._user_stopped:
                self.logMessage.emit("Tuning stopped by user")
                # Try to load any partial results that were saved
                if self._check_and_load_results():
                    self.logMessage.emit("Partial results saved successfully")
                else:
                    self.logMessage.emit("Waiting for results to be saved...")
                    # Give it a moment for files to be written
                    QTimer.singleShot(2000, self._delayed_result_check)
            else:
                self.logMessage.emit("Tuning process crashed unexpectedly")
                self.errorOccurred.emit("Tuning process crashed")
                
    def _delayed_result_check(self):
        """Check for results after a delay (for slow file writes)."""
        if self._check_and_load_results():
            self.logMessage.emit("Results saved successfully")
        else:
            self.logMessage.emit("No results found")
            
    @pyqtSlot(QProcess.ProcessError)
    def _handle_error(self, error: QProcess.ProcessError):
        """Handle process errors."""
        error_messages = {
            QProcess.ProcessError.FailedToStart: "Failed to start tuning process",
            QProcess.ProcessError.Crashed: "Tuning process crashed",
            QProcess.ProcessError.Timedout: "Tuning process timed out",
            QProcess.ProcessError.WriteError: "Error writing to tuning process",
            QProcess.ProcessError.ReadError: "Error reading from tuning process",
            QProcess.ProcessError.UnknownError: "Unknown error in tuning process"
        }
        
        msg = error_messages.get(error, "Unknown process error")
        self.errorOccurred.emit(msg)
        
    def is_running(self) -> bool:
        """Check if tuning is currently running."""
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning
    
    def _auto_stop_on_completion(self):
        """Automatically stop the process when tuning is complete."""
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            print("[INFO] Auto-stopping process after completion")
            self.logMessage.emit("Stopping process after successful completion...")
            self._process.terminate()
            if not self._process.waitForFinished(5000):
                self._process.kill()
                
    def _check_completion_timeout(self):
        """Check if the process is stuck after starting the final iteration."""
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            print("[WARNING] Process may be stuck after final iteration")
            self.logMessage.emit("Process appears stuck, attempting to save and exit...")
            self._process.terminate()
            if not self._process.waitForFinished(5000):
                self._process.kill()