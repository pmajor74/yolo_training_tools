"""Visualization tools for benchmark results including PR curves, charts, and plots."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import base64
from io import BytesIO
import traceback


class BenchmarkVisualizer:
    """Generate visualizations for benchmark results."""
    
    def __init__(self, class_names: Dict[int, str]):
        """Initialize visualizer with class names.
        
        Args:
            class_names: Dictionary mapping class IDs to names
        """
        # Ensure all keys are integers
        self.class_names = {}
        for k, v in class_names.items():
            try:
                self.class_names[int(k)] = v
            except (ValueError, TypeError):
                print(f"[WARNING] BenchmarkVisualizer: Skipping non-integer class ID: {k}")
                continue
                
        self.colors = self._generate_colors(len(self.class_names) + 1)
        
    def _generate_colors(self, n: int) -> List[str]:
        """Generate distinct colors for n classes."""
        colors = []
        for i in range(n):
            hue = i * 360 / n
            colors.append(f'hsl({hue}, 70%, 50%)')
        return colors
        
    def generate_pr_curves_html(self, per_class_metrics: Dict, overall_metrics: Dict) -> str:
        """Generate interactive Precision-Recall curves.
        
        Args:
            per_class_metrics: Per-class metrics with precision/recall data
            overall_metrics: Overall metrics
            
        Returns:
            HTML string with interactive PR curves
        """
        # Prepare data for plotting
        curves_data = []
        
        # Add overall curve
        if 'precision_recall_curve' in overall_metrics:
            curves_data.append({
                'name': 'Overall',
                'precision': overall_metrics['precision_recall_curve']['precision'],
                'recall': overall_metrics['precision_recall_curve']['recall'],
                'ap': overall_metrics.get('map_50', 0),
                'color': '#667eea',
                'width': 3
            })
            
        # Add per-class curves
        for i, (class_name, metrics) in enumerate(per_class_metrics.items()):
            if 'precision_recall_curve' in metrics:
                curves_data.append({
                    'name': class_name,
                    'precision': metrics['precision_recall_curve']['precision'],
                    'recall': metrics['precision_recall_curve']['recall'],
                    'ap': metrics.get('ap_by_iou', {}).get(0.5, 0),
                    'color': self.colors[i % len(self.colors)],
                    'width': 2
                })
                
        html = f"""
        <div class="pr-curves-container">
            <h3>Precision-Recall Curves</h3>
            <div class="curve-controls">
                <label>
                    <input type="checkbox" id="pr-show-all" checked>
                    Show All Classes
                </label>
                <label>
                    <input type="checkbox" id="pr-show-area">
                    Show Area Under Curve
                </label>
                <label>
                    <input type="checkbox" id="pr-show-grid" checked>
                    Show Grid
                </label>
            </div>
            <div id="pr-curves-chart"></div>
            <div id="pr-legend"></div>
        </div>
        
        <style>
            .pr-curves-container {{
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .curve-controls {{
                margin: 15px 0;
                padding: 10px;
                background: #f5f5f5;
                border-radius: 5px;
                display: flex;
                gap: 20px;
            }}
            #pr-curves-chart {{
                width: 100%;
                height: 500px;
                position: relative;
            }}
            #pr-legend {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-top: 15px;
                padding: 10px;
                background: #f9f9f9;
                border-radius: 5px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 5px;
                cursor: pointer;
                padding: 5px 10px;
                border-radius: 3px;
                transition: background 0.2s;
            }}
            .legend-item:hover {{
                background: #e0e0e0;
            }}
            .legend-item.disabled {{
                opacity: 0.4;
            }}
            .legend-color {{
                width: 20px;
                height: 3px;
                border-radius: 2px;
            }}
        </style>
        
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            (function() {{
                const curvesData = {json.dumps(curves_data)};
                let visibleCurves = new Set(curvesData.map((_, i) => i));
                
                function renderPRCurves() {{
                    const showArea = document.getElementById('pr-show-area').checked;
                    const showGrid = document.getElementById('pr-show-grid').checked;
                    
                    const traces = [];
                    
                    curvesData.forEach((curve, idx) => {{
                        if (!visibleCurves.has(idx)) return;
                        
                        // Main curve
                        traces.push({{
                            x: curve.recall,
                            y: curve.precision,
                            mode: 'lines',
                            name: `${{curve.name}} (AP: ${{(curve.ap * 100).toFixed(1)}}%)`,
                            line: {{
                                color: curve.color,
                                width: curve.width
                            }},
                            hovertemplate: '<b>${{curve.name}}</b><br>' +
                                         'Recall: %{{x:.3f}}<br>' +
                                         'Precision: %{{y:.3f}}<br>' +
                                         '<extra></extra>'
                        }});
                        
                        // Area under curve
                        if (showArea) {{
                            traces.push({{
                                x: curve.recall,
                                y: curve.precision,
                                fill: 'tozeroy',
                                fillcolor: curve.color.replace('hsl', 'hsla').replace(')', ', 0.1)'),
                                mode: 'none',
                                showlegend: false,
                                hoverinfo: 'skip'
                            }});
                        }}
                    }});
                    
                    const layout = {{
                        title: 'Precision-Recall Curves',
                        xaxis: {{
                            title: 'Recall',
                            range: [0, 1],
                            showgrid: showGrid,
                            gridcolor: '#e0e0e0'
                        }},
                        yaxis: {{
                            title: 'Precision',
                            range: [0, 1],
                            showgrid: showGrid,
                            gridcolor: '#e0e0e0'
                        }},
                        hovermode: 'closest',
                        plot_bgcolor: '#fafafa',
                        paper_bgcolor: 'white',
                        font: {{
                            family: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
                        }}
                    }};
                    
                    const config = {{
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d'],
                        toImageButtonOptions: {{
                            format: 'png',
                            filename: 'pr_curves',
                            width: 1200,
                            height: 800
                        }}
                    }};
                    
                    Plotly.newPlot('pr-curves-chart', traces, layout, config);
                }}
                
                function renderLegend() {{
                    const legendDiv = document.getElementById('pr-legend');
                    legendDiv.innerHTML = '';
                    
                    curvesData.forEach((curve, idx) => {{
                        const item = document.createElement('div');
                        item.className = 'legend-item' + (visibleCurves.has(idx) ? '' : ' disabled');
                        item.innerHTML = `
                            <div class="legend-color" style="background: ${{curve.color}}"></div>
                            <span>${{curve.name}} (AP: ${{(curve.ap * 100).toFixed(1)}}%)</span>
                        `;
                        item.addEventListener('click', () => {{
                            if (visibleCurves.has(idx)) {{
                                visibleCurves.delete(idx);
                            }} else {{
                                visibleCurves.add(idx);
                            }}
                            renderPRCurves();
                            renderLegend();
                        }});
                        legendDiv.appendChild(item);
                    }});
                }}
                
                // Event listeners
                document.getElementById('pr-show-all').addEventListener('change', (e) => {{
                    if (e.target.checked) {{
                        visibleCurves = new Set(curvesData.map((_, i) => i));
                    }} else {{
                        visibleCurves.clear();
                        visibleCurves.add(0); // Keep overall curve
                    }}
                    renderPRCurves();
                    renderLegend();
                }});
                
                document.getElementById('pr-show-area').addEventListener('change', renderPRCurves);
                document.getElementById('pr-show-grid').addEventListener('change', renderPRCurves);
                
                // Initial render
                renderPRCurves();
                renderLegend();
            }})();
        </script>
        """
        
        return html
        
    def generate_confidence_distribution_html(self, confidence_stats: Dict) -> str:
        """Generate confidence distribution histogram.
        
        Args:
            confidence_stats: Confidence statistics from benchmark
            
        Returns:
            HTML string with confidence distribution
        """
        html = f"""
        <div class="confidence-dist-container">
            <h3>Confidence Score Distribution</h3>
            <div id="confidence-histogram"></div>
            <div class="confidence-stats">
                <div class="stat-card">
                    <strong>TP Mean Confidence:</strong>
                    <span class="stat-value">{confidence_stats.get('tp_confidence_mean', 0):.3f}</span>
                </div>
                <div class="stat-card">
                    <strong>FP Mean Confidence:</strong>
                    <span class="stat-value">{confidence_stats.get('fp_confidence_mean', 0):.3f}</span>
                </div>
                <div class="stat-card">
                    <strong>Optimal Threshold:</strong>
                    <span class="stat-value">{confidence_stats.get('optimal_threshold', 0.25):.3f}</span>
                </div>
            </div>
        </div>
        
        <style>
            .confidence-dist-container {{
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            #confidence-histogram {{
                width: 100%;
                height: 400px;
            }}
            .confidence-stats {{
                display: flex;
                gap: 20px;
                margin-top: 20px;
                justify-content: center;
            }}
            .stat-card {{
                padding: 15px 20px;
                background: #f5f5f5;
                border-radius: 5px;
                text-align: center;
            }}
            .stat-value {{
                display: block;
                font-size: 1.5em;
                color: #667eea;
                margin-top: 5px;
                font-weight: bold;
            }}
        </style>
        
        <script>
            (function() {{
                const tpConfidences = {json.dumps(confidence_stats.get('tp_confidences', []))};
                const fpConfidences = {json.dumps(confidence_stats.get('fp_confidences', []))};
                
                const trace1 = {{
                    x: tpConfidences,
                    type: 'histogram',
                    name: 'True Positives',
                    marker: {{
                        color: '#4caf50'
                    }},
                    opacity: 0.7,
                    xbins: {{
                        start: 0,
                        end: 1,
                        size: 0.05
                    }}
                }};
                
                const trace2 = {{
                    x: fpConfidences,
                    type: 'histogram',
                    name: 'False Positives',
                    marker: {{
                        color: '#f44336'
                    }},
                    opacity: 0.7,
                    xbins: {{
                        start: 0,
                        end: 1,
                        size: 0.05
                    }}
                }};
                
                const layout = {{
                    title: 'Confidence Score Distribution',
                    xaxis: {{
                        title: 'Confidence Score',
                        range: [0, 1]
                    }},
                    yaxis: {{
                        title: 'Count'
                    }},
                    barmode: 'overlay',
                    hovermode: 'x',
                    plot_bgcolor: '#fafafa',
                    paper_bgcolor: 'white'
                }};
                
                const config = {{
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d']
                }};
                
                Plotly.newPlot('confidence-histogram', [trace1, trace2], layout, config);
            }})();
        </script>
        """
        
        return html
        
    def generate_f1_curve_html(self, thresholds: List[float], f1_scores: List[float]) -> str:
        """Generate F1 score vs confidence threshold curve.
        
        Args:
            thresholds: List of confidence thresholds
            f1_scores: Corresponding F1 scores
            
        Returns:
            HTML string with F1 curve
        """
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        html = f"""
        <div class="f1-curve-container">
            <h3>F1 Score vs Confidence Threshold</h3>
            <div id="f1-curve-chart"></div>
            <div class="optimal-point">
                <strong>Optimal Point:</strong> 
                Threshold = {optimal_threshold:.3f}, F1 = {optimal_f1:.3f}
            </div>
        </div>
        
        <style>
            .f1-curve-container {{
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            #f1-curve-chart {{
                width: 100%;
                height: 400px;
            }}
            .optimal-point {{
                text-align: center;
                margin-top: 15px;
                padding: 10px;
                background: #e8f5e9;
                border-radius: 5px;
                color: #2e7d32;
            }}
        </style>
        
        <script>
            (function() {{
                const trace = {{
                    x: {json.dumps(thresholds)},
                    y: {json.dumps(f1_scores)},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'F1 Score',
                    line: {{
                        color: '#667eea',
                        width: 3
                    }}
                }};
                
                const optimalPoint = {{
                    x: [{optimal_threshold}],
                    y: [{optimal_f1}],
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Optimal',
                    marker: {{
                        color: '#4caf50',
                        size: 12,
                        symbol: 'star'
                    }}
                }};
                
                const layout = {{
                    title: 'F1 Score vs Confidence Threshold',
                    xaxis: {{
                        title: 'Confidence Threshold',
                        range: [0, 1]
                    }},
                    yaxis: {{
                        title: 'F1 Score',
                        range: [0, 1]
                    }},
                    hovermode: 'closest',
                    plot_bgcolor: '#fafafa',
                    paper_bgcolor: 'white'
                }};
                
                const config = {{
                    responsive: true,
                    displayModeBar: true
                }};
                
                Plotly.newPlot('f1-curve-chart', [trace, optimalPoint], layout, config);
            }})();
        </script>
        """
        
        return html
        
    def generate_per_class_bar_chart_html(self, per_class_metrics: Dict, metric: str = 'ap_50') -> str:
        """Generate bar chart comparing per-class metrics.
        
        Args:
            per_class_metrics: Per-class metrics
            metric: Which metric to display
            
        Returns:
            HTML string with bar chart
        """
        class_names = []
        values = []
        colors = []
        
        for i, (class_name, metrics) in enumerate(per_class_metrics.items()):
            class_names.append(class_name)
            
            # Get the metric value
            if metric == 'ap_50':
                value = metrics.get('ap_by_iou', {}).get(0.5, 0)
            elif metric in ['precision', 'recall']:
                value = metrics.get('metrics_by_iou', {}).get(0.5, {}).get(metric, 0)
            else:
                value = 0
                
            values.append(value * 100)  # Convert to percentage
            
            # Color based on performance
            if value >= 0.8:
                colors.append('#4caf50')  # Green
            elif value >= 0.6:
                colors.append('#ff9800')  # Orange
            else:
                colors.append('#f44336')  # Red
                
        html = f"""
        <div class="class-bar-container">
            <h3>Per-Class Performance</h3>
            <div class="metric-selector">
                <label>
                    Metric:
                    <select id="class-metric-select">
                        <option value="ap_50" {'selected' if metric == 'ap_50' else ''}>AP@0.5</option>
                        <option value="precision" {'selected' if metric == 'precision' else ''}>Precision</option>
                        <option value="recall" {'selected' if metric == 'recall' else ''}>Recall</option>
                    </select>
                </label>
            </div>
            <div id="class-bar-chart"></div>
        </div>
        
        <style>
            .class-bar-container {{
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-selector {{
                margin: 10px 0;
                text-align: center;
            }}
            #class-bar-chart {{
                width: 100%;
                height: 400px;
            }}
        </style>
        
        <script>
            (function() {{
                const classNames = {json.dumps(class_names)};
                const values = {json.dumps(values)};
                const colors = {json.dumps(colors)};
                
                function renderBarChart() {{
                    const trace = {{
                        x: classNames,
                        y: values,
                        type: 'bar',
                        marker: {{
                            color: colors
                        }},
                        text: values.map(v => v.toFixed(1) + '%'),
                        textposition: 'outside',
                        hovertemplate: '<b>%{{x}}</b><br>%{{y:.1f}}%<extra></extra>'
                    }};
                    
                    const layout = {{
                        title: 'Per-Class {metric.upper().replace("_", "@")}',
                        xaxis: {{
                            title: 'Class',
                            tickangle: -45
                        }},
                        yaxis: {{
                            title: '{metric.upper().replace("_", "@")} (%)',
                            range: [0, Math.max(100, Math.max(...values) * 1.1)]
                        }},
                        plot_bgcolor: '#fafafa',
                        paper_bgcolor: 'white',
                        height: 400
                    }};
                    
                    const config = {{
                        responsive: true,
                        displayModeBar: true
                    }};
                    
                    Plotly.newPlot('class-bar-chart', [trace], layout, config);
                }}
                
                renderBarChart();
                
                // Note: Metric switching would require passing all metrics data
                // For now, it just displays the selected metric
            }})();
        </script>
        """
        
        return html
        
    def generate_iou_distribution_html(self, iou_distributions: Dict) -> str:
        """Generate IoU distribution visualization.
        
        Args:
            iou_distributions: IoU values for true positives
            
        Returns:
            HTML string with IoU distribution
        """
        html = f"""
        <div class="iou-dist-container">
            <h3>IoU Distribution for True Positives</h3>
            <div id="iou-histogram"></div>
            <div class="iou-stats">
                <p>Shows the distribution of Intersection over Union (IoU) values for correctly detected objects.</p>
                <p>Higher IoU values indicate better localization accuracy.</p>
            </div>
        </div>
        
        <style>
            .iou-dist-container {{
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            #iou-histogram {{
                width: 100%;
                height: 400px;
            }}
            .iou-stats {{
                margin-top: 15px;
                padding: 10px;
                background: #f5f5f5;
                border-radius: 5px;
                text-align: center;
                color: #666;
            }}
        </style>
        
        <script>
            (function() {{
                const iouValues = {json.dumps(iou_distributions.get('all_ious', []))};
                
                const trace = {{
                    x: iouValues,
                    type: 'histogram',
                    marker: {{
                        color: '#667eea'
                    }},
                    xbins: {{
                        start: 0,
                        end: 1,
                        size: 0.05
                    }},
                    hovertemplate: 'IoU: %{{x}}<br>Count: %{{y}}<extra></extra>'
                }};
                
                const layout = {{
                    title: 'IoU Distribution',
                    xaxis: {{
                        title: 'IoU Value',
                        range: [0, 1]
                    }},
                    yaxis: {{
                        title: 'Count'
                    }},
                    plot_bgcolor: '#fafafa',
                    paper_bgcolor: 'white',
                    shapes: [
                        {{
                            type: 'line',
                            x0: 0.5, x1: 0.5,
                            y0: 0, y1: 1,
                            yref: 'paper',
                            line: {{
                                color: 'red',
                                width: 2,
                                dash: 'dash'
                            }}
                        }},
                        {{
                            type: 'line',
                            x0: 0.75, x1: 0.75,
                            y0: 0, y1: 1,
                            yref: 'paper',
                            line: {{
                                color: 'orange',
                                width: 2,
                                dash: 'dash'
                            }}
                        }}
                    ],
                    annotations: [
                        {{
                            x: 0.5,
                            y: 1,
                            yref: 'paper',
                            text: 'IoU@0.5',
                            showarrow: false,
                            yanchor: 'bottom'
                        }},
                        {{
                            x: 0.75,
                            y: 1,
                            yref: 'paper',
                            text: 'IoU@0.75',
                            showarrow: false,
                            yanchor: 'bottom'
                        }}
                    ]
                }};
                
                const config = {{
                    responsive: true,
                    displayModeBar: true
                }};
                
                Plotly.newPlot('iou-histogram', [trace], layout, config);
            }})();
        </script>
        """
        
        return html