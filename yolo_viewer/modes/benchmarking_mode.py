"""Benchmarking mode for evaluating model performance against ground truth."""

import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QTextEdit, QProgressBar, QGroupBox,
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QFont

import numpy as np
from ultralytics import YOLO

from ..modes.base_mode import BaseMode
from ..core import ModelCache, DatasetManager
from ..utils.benchmark_calculator_advanced import AdvancedBenchmarkCalculator
from ..utils.benchmark_report_generator_v2 import EnhancedBenchmarkReportGenerator
from ..utils.benchmark_comparator import BenchmarkComparator


class BenchmarkWorker(QThread):
    """Worker thread for running benchmarks."""
    
    progress = pyqtSignal(int, str)  # Progress percentage and message
    result_ready = pyqtSignal(dict)  # Benchmark results
    error = pyqtSignal(str)  # Error message
    log_message = pyqtSignal(str)  # Log messages
    
    def __init__(self, model_path: str, test_folder: str, yaml_path: str, 
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        super().__init__()
        self.model_path = model_path
        self.test_folder = test_folder
        self.yaml_path = yaml_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._stop_requested = False
        
    def stop(self):
        """Request the worker to stop."""
        self._stop_requested = True
        
    def run(self):
        """Run the benchmark evaluation."""
        try:
            self.log_message.emit("[INFO] Starting advanced benchmark evaluation...")
            self.progress.emit(5, "Loading model...")
            
            # Load model
            model = YOLO(self.model_path)
            self.log_message.emit(f"[INFO] Model loaded: {Path(self.model_path).name}")
            
            # Load dataset configuration
            self.progress.emit(10, "Loading dataset configuration...")
            with open(self.yaml_path, 'r') as f:
                import yaml
                dataset_config = yaml.safe_load(f)
            
            class_names = dataset_config.get('names', {})
            self.log_message.emit(f"[INFO] Loaded {len(class_names)} classes from dataset config")
            
            # Get test images
            test_path = Path(self.test_folder)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            test_images = [f for f in test_path.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            if not test_images:
                raise ValueError(f"No images found in test folder: {self.test_folder}")
            
            self.log_message.emit(f"[INFO] Found {len(test_images)} test images")
            
            # Initialize advanced benchmark calculator
            calculator = AdvancedBenchmarkCalculator(class_names)
            
            # Process each image
            total_images = len(test_images)
            detailed_image_results = []
            
            for idx, image_path in enumerate(test_images):
                if self._stop_requested:
                    self.log_message.emit("[WARNING] Benchmark stopped by user")
                    return
                
                progress_pct = 15 + int((idx / total_images) * 70)
                self.progress.emit(progress_pct, f"Processing {image_path.name}...")
                self.log_message.emit(f"[INFO] Processing image {idx+1}/{total_images}: {image_path.name}")
                
                # Get ground truth annotations
                annotation_path = image_path.with_suffix('.txt')
                ground_truth = []
                
                if annotation_path.exists():
                    with open(annotation_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                ground_truth.append({
                                    'class_id': class_id,
                                    'bbox': [x_center, y_center, width, height],
                                    'confidence': 1.0
                                })
                
                # Run inference
                start_time = time.time()
                results = model(str(image_path), 
                              conf=self.conf_threshold,
                              iou=self.iou_threshold,
                              verbose=False)
                inference_time = time.time() - start_time
                
                # Extract predictions
                predictions = []
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            # Convert to normalized center format
                            img_h, img_w = result.orig_shape
                            x_center = ((x1 + x2) / 2) / img_w
                            y_center = ((y1 + y2) / 2) / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            
                            predictions.append({
                                'class_id': int(box.cls),
                                'bbox': [x_center, y_center, width, height],
                                'confidence': float(box.conf)
                            })
                
                # Calculate advanced metrics for this image
                image_metrics = calculator.calculate_image_metrics_advanced(
                    ground_truth, predictions, str(image_path)
                )
                image_metrics['inference_time'] = inference_time
                
                detailed_image_results.append(image_metrics)
                
                # Log summary for this image (using IoU 0.5 metrics)
                metrics_50 = image_metrics.get('metrics_by_iou', {}).get(0.5, {})
                self.log_message.emit(
                    f"  - GT: {len(ground_truth)}, Pred: {len(predictions)}, "
                    f"TP: {metrics_50.get('true_positives', 0)}, "
                    f"FP: {metrics_50.get('false_positives', 0)}, "
                    f"FN: {metrics_50.get('false_negatives', 0)}"
                )
            
            # Calculate overall advanced metrics
            self.progress.emit(85, "Calculating comprehensive metrics...")
            self.log_message.emit("[INFO] Calculating comprehensive benchmark metrics with COCO standards...")
            
            overall_metrics = calculator.calculate_overall_metrics_advanced(detailed_image_results)
            
            # Add metadata
            overall_metrics['metadata'] = {
                'model_path': self.model_path,
                'test_folder': self.test_folder,
                'yaml_path': self.yaml_path,
                'conf_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold,
                'num_images': total_images,
                'class_names': class_names,
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate average inference time
            total_inference_time = sum(r['inference_time'] for r in detailed_image_results)
            overall_metrics['total_inference_time'] = total_inference_time
            overall_metrics['avg_inference_time'] = total_inference_time / len(detailed_image_results) if detailed_image_results else 0
            
            # Store detailed results separately for advanced report
            overall_metrics['detailed_image_results'] = detailed_image_results
            
            self.progress.emit(100, "Benchmark complete!")
            self.log_message.emit("[SUCCESS] Advanced benchmark evaluation completed successfully")
            self.result_ready.emit(overall_metrics)
            
        except Exception as e:
            error_msg = f"Benchmark failed: {str(e)}"
            self.log_message.emit(f"[ERROR] {error_msg}")
            self.log_message.emit(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            self.error.emit(error_msg)


class BenchmarkingMode(BaseMode):
    """Benchmarking mode for model evaluation."""
    
    benchmarkStarted = pyqtSignal(str)  # Model path
    benchmarkCompleted = pyqtSignal(str)  # Report path
    benchmarkFailed = pyqtSignal(str)  # Error message
    statusMessage = pyqtSignal(str, int)  # Message and timeout
    
    def __init__(self):
        self._initialized = False
        super().__init__()
        self.model_cache = ModelCache()
        self.dataset_manager = DatasetManager()
        self.worker = None
        self.benchmark_results = None
        
        # Connect to model cache signals
        self.model_cache.modelLoaded.connect(lambda path: self._update_model_status())
        self.model_cache.modelCleared.connect(lambda: self._update_model_status())
        
    def _setup_ui(self):
        """Setup the benchmarking UI."""
        # Check if already initialized to avoid duplicate layouts
        if self._initialized:
            return
        self._initialized = True
        
        # Clear any existing layout
        if self.layout():
            QWidget().setLayout(self.layout())
        
        layout = QVBoxLayout(self)
        
        # Configuration section
        config_group = QGroupBox("Benchmark Configuration")
        config_layout = QVBoxLayout()
        
        # Model status
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Current Model:"))
        self.model_label = QLabel("No model loaded")
        self.model_label.setStyleSheet("QLabel { font-weight: bold; color: #667eea; }")
        model_layout.addWidget(self.model_label, 1)
        self.load_model_hint = QLabel("(Load model in Model Management tab)")
        self.load_model_hint.setStyleSheet("QLabel { color: #888; font-style: italic; }")
        model_layout.addWidget(self.load_model_hint)
        config_layout.addLayout(model_layout)
        
        # Test folder selection
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Test Folder:"))
        self.folder_edit = QLineEdit()
        folder_layout.addWidget(self.folder_edit, 1)
        self.browse_folder_btn = QPushButton("Browse...")
        self.browse_folder_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(self.browse_folder_btn)
        config_layout.addLayout(folder_layout)
        
        # YAML file selection
        yaml_layout = QHBoxLayout()
        yaml_layout.addWidget(QLabel("Dataset YAML:"))
        self.yaml_edit = QLineEdit()
        yaml_layout.addWidget(self.yaml_edit, 1)
        self.browse_yaml_btn = QPushButton("Browse...")
        self.browse_yaml_btn.clicked.connect(self._browse_yaml)
        yaml_layout.addWidget(self.browse_yaml_btn)
        config_layout.addLayout(yaml_layout)
        
        # Threshold settings
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        threshold_layout.addWidget(self.conf_spin)
        
        threshold_layout.addWidget(QLabel("IoU Threshold:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        threshold_layout.addWidget(self.iou_spin)
        threshold_layout.addStretch()
        config_layout.addLayout(threshold_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Benchmark")
        self.run_btn.clicked.connect(self._run_benchmark)
        self.run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        self.run_btn.setEnabled(False)  # Disabled until model is loaded
        control_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_benchmark)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self._export_report)
        self.export_btn.setEnabled(False)
        control_layout.addWidget(self.export_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Results section
        results_group = QGroupBox("Results Summary")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Log section
        log_group = QGroupBox("Benchmark Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
    def _on_activate(self):
        """Called when this mode is activated."""
        # Setup UI if not already done
        if not self._initialized:
            self._setup_ui()
        self._update_model_status()
        if hasattr(self, 'log_text'):
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Benchmarking mode activated")
        self.statusMessage.emit("Benchmarking Mode - Evaluate model performance", 3000)
        
    def _on_deactivate(self) -> Optional[bool]:
        """Called when switching away from this mode."""
        if self.worker and self.worker.isRunning():
            # Ask user if they want to stop the benchmark
            reply = QMessageBox.question(
                self, "Stop Benchmark?",
                "A benchmark is currently running. Do you want to stop it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.worker.wait()
                return True
            else:
                return False  # Cancel tab switch
        return True  # Allow tab switch
    
    def get_mode_name(self) -> str:
        """Get the display name of this mode."""
        return "Benchmarking"
            
    def _update_model_status(self):
        """Update the model status display."""
        if not hasattr(self, 'model_label'):
            return  # UI not initialized yet
            
        if self.model_cache.is_loaded():
            model_path = self.model_cache.get_model_path()
            if model_path:
                model_name = Path(model_path).name
                self.model_label.setText(model_name)
                self.model_label.setStyleSheet("QLabel { font-weight: bold; color: #4caf50; }")
                self.load_model_hint.hide()
                self.run_btn.setEnabled(True)
        else:
            self.model_label.setText("No model loaded")
            self.model_label.setStyleSheet("QLabel { font-weight: bold; color: #f44336; }")
            self.load_model_hint.show()
            self.run_btn.setEnabled(False)
            
    def _browse_folder(self):
        """Browse for test folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Test Folder"
        )
        if folder:
            self.folder_edit.setText(folder)
            
    def _browse_yaml(self):
        """Browse for dataset YAML file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset YAML", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.yaml_edit.setText(file_path)
            
    @pyqtSlot()
    def _run_benchmark(self):
        """Run the benchmark evaluation."""
        # Check if model is loaded
        if not self.model_cache.is_loaded():
            QMessageBox.warning(self, "Warning", "Please load a model in the Model Management tab first")
            return
            
        # Get model path from cache
        model_path = self.model_cache.get_model_path()
        if not model_path:
            QMessageBox.warning(self, "Warning", "Model path not available")
            return
            
        # Validate inputs
        test_folder = self.folder_edit.text()
        yaml_path = self.yaml_edit.text()
            
        if not test_folder or not Path(test_folder).exists():
            QMessageBox.warning(self, "Warning", "Please select a valid test folder")
            return
            
        if not yaml_path or not Path(yaml_path).exists():
            QMessageBox.warning(self, "Warning", "Please select a valid dataset YAML file")
            return
            
                
        # Clear previous results
        self.results_table.setRowCount(0)
        self.log_text.clear()
        self.benchmark_results = None
        
        # Create and start worker
        self.worker = BenchmarkWorker(
            model_path,
            test_folder,
            yaml_path,
            self.conf_spin.value(),
            self.iou_spin.value()
        )
        
        # Connect signals
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_results_ready)
        self.worker.error.connect(self._on_error)
        self.worker.log_message.connect(self._on_log_message)
        
        # Update UI
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting benchmark...")
        
        # Start worker
        self.worker.start()
        self.benchmarkStarted.emit(model_path)
        
    @pyqtSlot()
    def _stop_benchmark(self):
        """Stop the running benchmark."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Stopping...")
            self.stop_btn.setEnabled(False)
            
    @pyqtSlot(int, str)
    def _on_progress(self, value: int, message: str):
        """Update progress display."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
    @pyqtSlot(str)
    def _on_log_message(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        
    @pyqtSlot(dict)
    def _on_results_ready(self, results: dict):
        """Handle benchmark results."""
        self.benchmark_results = results
        
        # Update UI
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(True)
        
        # Display summary metrics
        self._display_results(results)
        
        # Auto-export report
        report_path = self._auto_export_report(results)
        if report_path:
            self.benchmarkCompleted.emit(report_path)
            self.statusMessage.emit(f"Report saved to: {report_path}", 5000)
            
    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        """Handle benchmark error."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Error occurred")
        QMessageBox.critical(self, "Benchmark Error", error_msg)
        self.benchmarkFailed.emit(error_msg)
        
    def _display_results(self, results: dict):
        """Display results in the table."""
        self.results_table.setRowCount(0)
        
        # Key metrics to display - updated for advanced metrics
        display_metrics = [
            ('Total Images', results.get('total_images', 0)),
            ('Total Ground Truth', results.get('total_ground_truth', 0)),
            ('Total Predictions', results.get('total_predictions', 0)),
            ('True Positives', results.get('true_positives', 0)),
            ('False Positives', results.get('false_positives', 0)),
            ('False Negatives', results.get('false_negatives', 0)),
            ('Precision', f"{results.get('precision', 0):.4f}"),
            ('Recall', f"{results.get('recall', 0):.4f}"),
            ('F1 Score', f"{results.get('f1_score', 0):.4f}"),
            ('mAP@0.5', f"{results.get('map_50', 0):.4f}"),
            ('mAP@0.75', f"{results.get('map_75', 0):.4f}"),
            ('mAP@[0.5:0.95]', f"{results.get('map_50_95', 0):.4f}"),
            ('Avg Inference Time', f"{results.get('avg_inference_time', 0):.4f}s"),
        ]
        
        # Add per-class metrics if available (limited to top 5 classes)
        if 'per_class_metrics' in results:
            class_items = list(results['per_class_metrics'].items())[:5]
            for class_name, metrics in class_items:
                # Get metrics at IoU 0.5
                ap_50 = metrics.get('ap_by_iou', {}).get(0.5, 0)
                metrics_50 = metrics.get('metrics_by_iou', {}).get(0.5, {})
                display_metrics.extend([
                    (f'{class_name} AP@0.5', f"{ap_50:.4f}"),
                    (f'{class_name} Precision', f"{metrics_50.get('precision', 0):.4f}"),
                    (f'{class_name} Recall', f"{metrics_50.get('recall', 0):.4f}"),
                ])
        
        # Populate table
        for metric_name, value in display_metrics:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(metric_name))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(value)))
            
    def _auto_export_report(self, results: dict) -> Optional[str]:
        """Automatically export the enhanced report to the test folder."""
        try:
            test_folder = Path(self.folder_edit.text())
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_name = f"benchmark_report_enhanced_{timestamp}.html"
            report_path = test_folder / report_name
            
            # Generate enhanced HTML report
            generator = EnhancedBenchmarkReportGenerator()
            detailed_results = results.get('detailed_image_results', [])
            html_content = generator.generate_report(results, detailed_results)
            
            # Save report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self._on_log_message(f"[SUCCESS] Enhanced report saved to: {report_path}")
            
            return str(report_path)
            
        except Exception as e:
            self._on_log_message(f"[ERROR] Failed to save report: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    @pyqtSlot()
    def _export_report(self):
        """Export benchmark report to HTML."""
        if not self.benchmark_results:
            QMessageBox.warning(self, "Warning", "No benchmark results to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Enhanced Benchmark Report", 
            f"benchmark_report_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html)"
        )
        
        if file_path:
            try:
                generator = EnhancedBenchmarkReportGenerator()
                detailed_results = self.benchmark_results.get('detailed_image_results', [])
                html_content = generator.generate_report(self.benchmark_results, detailed_results)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
                QMessageBox.information(self, "Success", f"Enhanced report saved to:\n{file_path}")
                self.statusMessage.emit(f"Enhanced report exported to: {Path(file_path).name}", 5000)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export report:\n{str(e)}")
                import traceback
                traceback.print_exc()