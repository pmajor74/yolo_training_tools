"""Confusion matrix generator for object detection with interactive visualization."""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json


class ConfusionMatrixGenerator:
    """Generate and visualize confusion matrices for object detection."""
    
    def __init__(self, class_names: Dict[int, str]):
        """Initialize with class names.
        
        Args:
            class_names: Dictionary mapping class IDs to names
        """
        # Ensure all keys are integers
        self.class_names = {}
        for k, v in class_names.items():
            try:
                self.class_names[int(k)] = v
            except (ValueError, TypeError):
                # If key cannot be converted to int, skip it
                print(f"[WARNING] ConfusionMatrixGenerator: Skipping non-integer class ID: {k}")
                continue
                
        self.num_classes = len(self.class_names)
        
        # Create reverse mapping for convenience
        self.name_to_id = {name: id for id, name in self.class_names.items()}
        
        # Extended class names including background
        # Sort by key to ensure consistent ordering
        sorted_names = [self.class_names[i] for i in sorted(self.class_names.keys())]
        self.extended_class_names = sorted_names + ['Background/Missed']
        
    def generate_html_heatmap(self, confusion_matrix: np.ndarray, 
                             normalize: bool = True,
                             title: str = "Confusion Matrix") -> str:
        """Generate interactive HTML heatmap for confusion matrix.
        
        Args:
            confusion_matrix: The confusion matrix to visualize
            normalize: Whether to normalize by row (true class)
            title: Title for the visualization
            
        Returns:
            HTML string with interactive heatmap
        """
        # Normalize if requested
        if normalize:
            cm_normalized = confusion_matrix.astype('float')
            row_sums = cm_normalized.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            cm_normalized = cm_normalized / row_sums
        else:
            cm_normalized = confusion_matrix
            
        # Convert to list for JSON serialization
        cm_data = cm_normalized.tolist()
        cm_raw = confusion_matrix.tolist()
        
        # Generate HTML with embedded JavaScript for interactivity
        html = f"""
        <div class="confusion-matrix-container">
            <h3>{title}</h3>
            <div class="matrix-controls">
                <label>
                    <input type="checkbox" id="cm-normalize" {'checked' if normalize else ''}>
                    Normalize by True Class
                </label>
                <label>
                    <input type="checkbox" id="cm-show-zeros">
                    Hide Zero Values
                </label>
                <label>
                    Color Scheme:
                    <select id="cm-colorscheme">
                        <option value="viridis">Viridis</option>
                        <option value="plasma">Plasma</option>
                        <option value="blues" selected>Blues</option>
                        <option value="reds">Reds</option>
                    </select>
                </label>
            </div>
            <div id="confusion-matrix-heatmap"></div>
            <div id="cm-details" class="matrix-details"></div>
        </div>
        
        <style>
            .confusion-matrix-container {{
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .matrix-controls {{
                margin: 15px 0;
                padding: 10px;
                background: #f5f5f5;
                border-radius: 5px;
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }}
            .matrix-controls label {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            .matrix-cell {{
                cursor: pointer;
                stroke: white;
                stroke-width: 2;
            }}
            .matrix-cell:hover {{
                stroke: #333;
                stroke-width: 3;
            }}
            .matrix-details {{
                margin-top: 20px;
                padding: 15px;
                background: #f9f9f9;
                border-radius: 5px;
                min-height: 60px;
            }}
            .detail-card {{
                display: inline-block;
                margin: 5px;
                padding: 8px 12px;
                background: white;
                border-radius: 4px;
                border: 1px solid #ddd;
            }}
            .cell-tooltip {{
                position: absolute;
                background: rgba(0,0,0,0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
            }}
        </style>
        
        <script>
            (function() {{
                const rawMatrix = {json.dumps(cm_raw)};
                const classNames = {json.dumps(self.extended_class_names)};
                let currentMatrix = {json.dumps(cm_data)};
                let normalized = {str(normalize).lower()};
                
                function getColor(value, scheme) {{
                    // Color schemes
                    const schemes = {{
                        'viridis': ['#440154', '#31688e', '#35b779', '#fde725'],
                        'plasma': ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636'],
                        'blues': ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#084594'],
                        'reds': ['#fff5f0', '#fcbba1', '#fb6a4a', '#cb181d', '#67000d']
                    }};
                    
                    const colors = schemes[scheme] || schemes['blues'];
                    const idx = Math.floor(value * (colors.length - 1));
                    return colors[Math.min(idx, colors.length - 1)];
                }}
                
                function renderMatrix() {{
                    const container = document.getElementById('confusion-matrix-heatmap');
                    const hideZeros = document.getElementById('cm-show-zeros').checked;
                    const colorScheme = document.getElementById('cm-colorscheme').value;
                    
                    const cellSize = 40;
                    const labelWidth = 150;
                    const labelHeight = 150;
                    const width = cellSize * classNames.length + labelWidth;
                    const height = cellSize * classNames.length + labelHeight;
                    
                    // Clear previous content
                    container.innerHTML = '';
                    
                    // Create SVG
                    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                    svg.setAttribute('width', width);
                    svg.setAttribute('height', height);
                    
                    // Find max value for color scaling
                    const maxValue = Math.max(...currentMatrix.flat());
                    
                    // Draw cells
                    for (let i = 0; i < classNames.length; i++) {{
                        for (let j = 0; j < classNames.length; j++) {{
                            const value = currentMatrix[i][j];
                            const rawValue = rawMatrix[i][j];
                            
                            if (hideZeros && rawValue === 0) continue;
                            
                            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                            rect.setAttribute('x', j * cellSize + labelWidth);
                            rect.setAttribute('y', i * cellSize + labelHeight);
                            rect.setAttribute('width', cellSize);
                            rect.setAttribute('height', cellSize);
                            rect.setAttribute('class', 'matrix-cell');
                            rect.setAttribute('fill', getColor(value / (maxValue || 1), colorScheme));
                            rect.setAttribute('data-true', i);
                            rect.setAttribute('data-pred', j);
                            rect.setAttribute('data-value', value);
                            rect.setAttribute('data-raw', rawValue);
                            
                            // Add click handler
                            rect.addEventListener('click', function() {{
                                showCellDetails(i, j, rawValue, value);
                            }});
                            
                            // Add hover tooltip
                            rect.addEventListener('mouseenter', function(e) {{
                                const tooltip = document.createElement('div');
                                tooltip.className = 'cell-tooltip';
                                tooltip.innerHTML = `
                                    <strong>True: ${{classNames[i]}}</strong><br>
                                    <strong>Predicted: ${{classNames[j]}}</strong><br>
                                    Count: ${{rawValue}}<br>
                                    ${{normalized ? `Percentage: ${{(value * 100).toFixed(1)}}%` : ''}}
                                `;
                                tooltip.style.left = e.pageX + 10 + 'px';
                                tooltip.style.top = e.pageY + 10 + 'px';
                                document.body.appendChild(tooltip);
                                rect.tooltip = tooltip;
                            }});
                            
                            rect.addEventListener('mouseleave', function() {{
                                if (rect.tooltip) {{
                                    rect.tooltip.remove();
                                }}
                            }});
                            
                            svg.appendChild(rect);
                            
                            // Add text label if value is significant
                            if (rawValue > 0) {{
                                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                                text.setAttribute('x', j * cellSize + labelWidth + cellSize/2);
                                text.setAttribute('y', i * cellSize + labelHeight + cellSize/2);
                                text.setAttribute('text-anchor', 'middle');
                                text.setAttribute('dominant-baseline', 'middle');
                                text.setAttribute('fill', value > 0.5 ? 'white' : 'black');
                                text.setAttribute('font-size', '10');
                                text.setAttribute('pointer-events', 'none');
                                text.textContent = normalized ? 
                                    (value * 100).toFixed(0) + '%' : 
                                    rawValue.toString();
                                svg.appendChild(text);
                            }}
                        }}
                    }}
                    
                    // Draw labels
                    classNames.forEach((name, idx) => {{
                        // Y-axis labels (True class)
                        const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                        yLabel.setAttribute('x', labelWidth - 5);
                        yLabel.setAttribute('y', idx * cellSize + labelHeight + cellSize/2);
                        yLabel.setAttribute('text-anchor', 'end');
                        yLabel.setAttribute('dominant-baseline', 'middle');
                        yLabel.setAttribute('font-size', '11');
                        yLabel.textContent = name;
                        svg.appendChild(yLabel);
                        
                        // X-axis labels (Predicted class)
                        const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                        xLabel.setAttribute('x', idx * cellSize + labelWidth + cellSize/2);
                        xLabel.setAttribute('y', labelHeight - 5);
                        xLabel.setAttribute('text-anchor', 'middle');
                        xLabel.setAttribute('font-size', '11');
                        xLabel.setAttribute('transform', 
                            `rotate(-45, ${{idx * cellSize + labelWidth + cellSize/2}}, ${{labelHeight - 5}})`);
                        xLabel.textContent = name;
                        svg.appendChild(xLabel);
                    }});
                    
                    // Add axis labels
                    const yAxisLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    yAxisLabel.setAttribute('x', 20);
                    yAxisLabel.setAttribute('y', height/2);
                    yAxisLabel.setAttribute('text-anchor', 'middle');
                    yAxisLabel.setAttribute('font-size', '14');
                    yAxisLabel.setAttribute('font-weight', 'bold');
                    yAxisLabel.setAttribute('transform', `rotate(-90, 20, ${{height/2}})`);
                    yAxisLabel.textContent = 'True Class';
                    svg.appendChild(yAxisLabel);
                    
                    const xAxisLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    xAxisLabel.setAttribute('x', width/2);
                    xAxisLabel.setAttribute('y', 20);
                    xAxisLabel.setAttribute('text-anchor', 'middle');
                    xAxisLabel.setAttribute('font-size', '14');
                    xAxisLabel.setAttribute('font-weight', 'bold');
                    xAxisLabel.textContent = 'Predicted Class';
                    svg.appendChild(xAxisLabel);
                    
                    container.appendChild(svg);
                }}
                
                function showCellDetails(trueIdx, predIdx, rawCount, normValue) {{
                    const detailsDiv = document.getElementById('cm-details');
                    const trueName = classNames[trueIdx];
                    const predName = classNames[predIdx];
                    
                    let interpretation = '';
                    if (trueIdx === predIdx && trueIdx < classNames.length - 1) {{
                        interpretation = `<span style="color: green;">✓ Correct Detection</span>`;
                    }} else if (trueIdx === classNames.length - 1 && predIdx < classNames.length - 1) {{
                        interpretation = `<span style="color: orange;">⚠ False Positive (detected ${{predName}} on background)</span>`;
                    }} else if (predIdx === classNames.length - 1 && trueIdx < classNames.length - 1) {{
                        interpretation = `<span style="color: red;">✗ False Negative (missed ${{trueName}})</span>`;
                    }} else if (trueIdx !== predIdx) {{
                        interpretation = `<span style="color: orange;">⚠ Misclassification (${{trueName}} detected as ${{predName}})</span>`;
                    }}
                    
                    detailsDiv.innerHTML = `
                        <h4>Cell Details</h4>
                        <div class="detail-card">
                            <strong>True Class:</strong> ${{trueName}}
                        </div>
                        <div class="detail-card">
                            <strong>Predicted Class:</strong> ${{predName}}
                        </div>
                        <div class="detail-card">
                            <strong>Count:</strong> ${{rawCount}}
                        </div>
                        ${{normalized ? `<div class="detail-card"><strong>Percentage:</strong> ${{(normValue * 100).toFixed(2)}}%</div>` : ''}}
                        <div style="margin-top: 10px;">
                            ${{interpretation}}
                        </div>
                    `;
                }}
                
                function updateMatrix() {{
                    normalized = document.getElementById('cm-normalize').checked;
                    
                    if (normalized) {{
                        // Normalize by row
                        currentMatrix = rawMatrix.map(row => {{
                            const sum = row.reduce((a, b) => a + b, 0);
                            return sum > 0 ? row.map(val => val / sum) : row;
                        }});
                    }} else {{
                        currentMatrix = rawMatrix.map(row => [...row]);
                    }}
                    
                    renderMatrix();
                }}
                
                // Set up event listeners
                document.getElementById('cm-normalize').addEventListener('change', updateMatrix);
                document.getElementById('cm-show-zeros').addEventListener('change', renderMatrix);
                document.getElementById('cm-colorscheme').addEventListener('change', renderMatrix);
                
                // Initial render
                renderMatrix();
                
                // Show summary in details
                const totalCorrect = rawMatrix.reduce((sum, row, i) => 
                    sum + (i < classNames.length - 1 ? row[i] : 0), 0);
                const totalSamples = rawMatrix.flat().reduce((a, b) => a + b, 0);
                const accuracy = totalSamples > 0 ? (totalCorrect / totalSamples * 100).toFixed(1) : 0;
                
                document.getElementById('cm-details').innerHTML = `
                    <h4>Matrix Summary</h4>
                    <div class="detail-card">
                        <strong>Total Samples:</strong> ${{totalSamples}}
                    </div>
                    <div class="detail-card">
                        <strong>Correct Detections:</strong> ${{totalCorrect}}
                    </div>
                    <div class="detail-card">
                        <strong>Overall Accuracy:</strong> ${{accuracy}}%
                    </div>
                    <p style="margin-top: 10px; color: #666;">
                        Click on any cell for detailed information
                    </p>
                `;
            }})();
        </script>
        """
        
        return html
        
    def generate_class_confusion_analysis(self, confusion_matrix: np.ndarray) -> Dict:
        """Analyze confusion patterns for each class.
        
        Returns:
            Dictionary with confusion analysis per class
        """
        analysis = {}
        
        for class_id, class_name in self.class_names.items():
            # Ensure class_id is an integer
            class_idx = int(class_id)
            
            # Skip if index is out of bounds
            if class_idx >= confusion_matrix.shape[0] - 1:  # -1 for background row
                continue
                
            # Get row for this class (true labels)
            row = confusion_matrix[class_idx]
            
            # Get column for this class (predictions)
            col = confusion_matrix[:, class_idx]
            
            # Calculate metrics
            true_positives = row[class_idx]
            false_negatives = np.sum(row) - true_positives
            false_positives = np.sum(col) - true_positives
            
            # Find main confusion partners
            confused_with = []
            for other_id, value in enumerate(row):
                if other_id != class_idx and value > 0:
                    other_name = (self.class_names.get(other_id, 'Background') 
                                 if other_id < self.num_classes else 'Background/Missed')
                    confused_with.append({
                        'class': other_name,
                        'count': int(value),
                        'percentage': float(value / np.sum(row) * 100) if np.sum(row) > 0 else 0
                    })
                    
            confused_with.sort(key=lambda x: x['count'], reverse=True)
            
            analysis[class_name] = {
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives),
                'precision': float(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0,
                'recall': float(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0,
                'confused_with': confused_with[:5]  # Top 5 confusions
            }
            
        return analysis
        
    def generate_summary_metrics(self, confusion_matrix: np.ndarray) -> Dict:
        """Generate summary metrics from confusion matrix.
        
        Returns:
            Dictionary with overall metrics
        """
        # Overall accuracy (excluding background)
        correct = np.trace(confusion_matrix[:-1, :-1])
        total = np.sum(confusion_matrix[:-1, :])
        
        # Per-class accuracies
        per_class_accuracy = []
        for i in range(self.num_classes):
            row_sum = np.sum(confusion_matrix[i, :])
            if row_sum > 0:
                accuracy = confusion_matrix[i, i] / row_sum
                per_class_accuracy.append(accuracy)
                
        # Background/Missed statistics
        total_fps = np.sum(confusion_matrix[-1, :-1])  # Background predicted as classes
        total_fns = np.sum(confusion_matrix[:-1, -1])  # Classes predicted as background
        
        return {
            'overall_accuracy': float(correct / total) if total > 0 else 0,
            'mean_class_accuracy': float(np.mean(per_class_accuracy)) if per_class_accuracy else 0,
            'total_false_positives': int(total_fps),
            'total_false_negatives': int(total_fns),
            'total_correct': int(correct),
            'total_samples': int(total)
        }