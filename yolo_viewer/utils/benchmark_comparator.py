"""Multi-model comparison tools for benchmark results."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path


class BenchmarkComparator:
    """Compare multiple model benchmark results."""
    
    def __init__(self, class_names: Dict[int, str]):
        """Initialize comparator with class names.
        
        Args:
            class_names: Dictionary mapping class IDs to names
        """
        # Ensure all keys are integers
        self.class_names = {}
        for k, v in class_names.items():
            try:
                self.class_names[int(k)] = v
            except (ValueError, TypeError):
                print(f"[WARNING] BenchmarkComparator: Skipping non-integer class ID: {k}")
                continue
                
        self.models_data = {}
        
    def add_model_results(self, model_name: str, results: Dict):
        """Add benchmark results for a model.
        
        Args:
            model_name: Name/identifier for the model
            results: Complete benchmark results dictionary
        """
        self.models_data[model_name] = results
        
    def compare_overall_metrics(self) -> Dict:
        """Compare overall metrics across models.
        
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            'models': [],
            'metrics': {}
        }
        
        metric_keys = [
            'map_50', 'map_75', 'map_50_95',
            'precision', 'recall', 'f1_score',
            'true_positives', 'false_positives', 'false_negatives',
            'avg_inference_time'
        ]
        
        for metric in metric_keys:
            comparison['metrics'][metric] = []
            
        for model_name, results in self.models_data.items():
            comparison['models'].append(model_name)
            
            for metric in metric_keys:
                if metric == 'avg_inference_time':
                    value = results.get('avg_inference_time', 0)
                else:
                    value = results.get(metric, 0)
                comparison['metrics'][metric].append(value)
                
        # Find best model for each metric
        comparison['best_models'] = {}
        for metric, values in comparison['metrics'].items():
            if values:
                if metric in ['false_positives', 'false_negatives', 'avg_inference_time']:
                    # Lower is better
                    best_idx = np.argmin(values)
                else:
                    # Higher is better
                    best_idx = np.argmax(values)
                comparison['best_models'][metric] = comparison['models'][best_idx]
                
        return comparison
        
    def compare_per_class_performance(self) -> Dict:
        """Compare per-class performance across models.
        
        Returns:
            Dictionary with per-class comparison
        """
        comparison = {}
        
        # Collect all class names
        all_classes = set()
        for results in self.models_data.values():
            per_class = results.get('per_class_metrics', {})
            all_classes.update(per_class.keys())
            
        for class_name in sorted(all_classes):
            comparison[class_name] = {
                'models': [],
                'ap_50': [],
                'precision': [],
                'recall': []
            }
            
            for model_name, results in self.models_data.items():
                per_class = results.get('per_class_metrics', {})
                if class_name in per_class:
                    class_metrics = per_class[class_name]
                    comparison[class_name]['models'].append(model_name)
                    comparison[class_name]['ap_50'].append(
                        class_metrics.get('ap_by_iou', {}).get(0.5, 0)
                    )
                    comparison[class_name]['precision'].append(
                        class_metrics.get('metrics_by_iou', {}).get(0.5, {}).get('precision', 0)
                    )
                    comparison[class_name]['recall'].append(
                        class_metrics.get('metrics_by_iou', {}).get(0.5, {}).get('recall', 0)
                    )
                else:
                    comparison[class_name]['models'].append(model_name)
                    comparison[class_name]['ap_50'].append(0)
                    comparison[class_name]['precision'].append(0)
                    comparison[class_name]['recall'].append(0)
                    
        return comparison
        
    def generate_comparison_html(self) -> str:
        """Generate HTML comparison report.
        
        Returns:
            HTML string with comparison visualizations
        """
        overall_comp = self.compare_overall_metrics()
        class_comp = self.compare_per_class_performance()
        
        html = f"""
        <div class="model-comparison-container">
            <h2>Multi-Model Comparison</h2>
            
            {self._generate_overall_comparison_section(overall_comp)}
            {self._generate_radar_chart_section(overall_comp)}
            {self._generate_per_class_comparison_section(class_comp)}
            {self._generate_speed_vs_accuracy_section(overall_comp)}
            {self._generate_recommendation_section(overall_comp)}
        </div>
        
        <style>
            .model-comparison-container {{
                padding: 20px;
                background: white;
            }}
            .comparison-section {{
                margin: 30px 0;
                padding: 20px;
                background: #f9f9f9;
                border-radius: 8px;
            }}
            .comparison-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            .comparison-table th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            .comparison-table td {{
                padding: 10px;
                border-bottom: 1px solid #e0e0e0;
            }}
            .comparison-table tr:nth-child(even) {{
                background: white;
            }}
            .best-value {{
                font-weight: bold;
                color: #4caf50;
                position: relative;
            }}
            .best-value::after {{
                content: " ⭐";
            }}
            .worst-value {{
                color: #f44336;
            }}
            .chart-container {{
                width: 100%;
                height: 500px;
                margin: 20px 0;
            }}
        </style>
        """
        
        return html
        
    def _generate_overall_comparison_section(self, comparison: Dict) -> str:
        """Generate overall metrics comparison table."""
        models = comparison['models']
        metrics = comparison['metrics']
        best_models = comparison['best_models']
        
        # Create table rows
        rows = []
        metric_names = {
            'map_50': 'mAP@0.5',
            'map_75': 'mAP@0.75',
            'map_50_95': 'mAP@[0.5:0.95]',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1 Score',
            'true_positives': 'True Positives',
            'false_positives': 'False Positives',
            'false_negatives': 'False Negatives',
            'avg_inference_time': 'Avg Inference Time'
        }
        
        for metric_key, metric_name in metric_names.items():
            if metric_key not in metrics:
                continue
                
            row = f"<tr><td><strong>{metric_name}</strong></td>"
            values = metrics[metric_key]
            
            for i, model in enumerate(models):
                value = values[i]
                is_best = best_models.get(metric_key) == model
                
                # Format value
                if metric_key in ['map_50', 'map_75', 'map_50_95', 'precision', 'recall', 'f1_score']:
                    formatted = f"{value:.1%}"
                elif metric_key == 'avg_inference_time':
                    formatted = f"{value:.3f}s"
                else:
                    formatted = str(int(value))
                    
                css_class = "best-value" if is_best else ""
                row += f'<td class="{css_class}">{formatted}</td>'
                
            row += "</tr>"
            rows.append(row)
            
        # Create header
        header = "<tr><th>Metric</th>"
        for model in models:
            header += f"<th>{model}</th>"
        header += "</tr>"
        
        return f"""
        <div class="comparison-section">
            <h3>Overall Performance Metrics</h3>
            <table class="comparison-table">
                <thead>{header}</thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """
        
    def _generate_radar_chart_section(self, comparison: Dict) -> str:
        """Generate radar chart for multi-metric comparison."""
        models = comparison['models']
        
        # Select key metrics for radar chart
        radar_metrics = ['map_50', 'precision', 'recall', 'f1_score']
        
        traces_data = []
        for i, model in enumerate(models):
            values = []
            for metric in radar_metrics:
                value = comparison['metrics'][metric][i]
                values.append(value * 100)  # Convert to percentage
                
            traces_data.append({
                'name': model,
                'values': values
            })
            
        return f"""
        <div class="comparison-section">
            <h3>Performance Radar Chart</h3>
            <div id="radar-chart" class="chart-container"></div>
        </div>
        
        <script>
            (function() {{
                const models = {json.dumps(models)};
                const metrics = ['mAP@0.5', 'Precision', 'Recall', 'F1 Score'];
                const tracesData = {json.dumps(traces_data)};
                
                const traces = tracesData.map((data, idx) => ({{
                    type: 'scatterpolar',
                    r: data.values,
                    theta: metrics,
                    fill: 'toself',
                    name: data.name,
                    line: {{
                        width: 2
                    }},
                    marker: {{
                        size: 8
                    }}
                }}));
                
                const layout = {{
                    polar: {{
                        radialaxis: {{
                            visible: true,
                            range: [0, 100],
                            ticksuffix: '%'
                        }}
                    }},
                    showlegend: true,
                    title: 'Model Performance Comparison',
                    height: 500
                }};
                
                Plotly.newPlot('radar-chart', traces, layout, {{responsive: true}});
            }})();
        </script>
        """
        
    def _generate_per_class_comparison_section(self, comparison: Dict) -> str:
        """Generate per-class comparison visualization."""
        # Select top classes by variance in performance
        class_variances = {}
        for class_name, data in comparison.items():
            if data['ap_50']:
                variance = np.var(data['ap_50'])
                class_variances[class_name] = variance
                
        # Sort by variance and take top 10
        top_classes = sorted(class_variances.items(), key=lambda x: x[1], reverse=True)[:10]
        top_class_names = [c[0] for c in top_classes]
        
        # Prepare data for grouped bar chart
        models = list(self.models_data.keys())
        chart_data = []
        
        for model in models:
            ap_values = []
            for class_name in top_class_names:
                if model in comparison[class_name]['models']:
                    idx = comparison[class_name]['models'].index(model)
                    ap_values.append(comparison[class_name]['ap_50'][idx] * 100)
                else:
                    ap_values.append(0)
            chart_data.append({
                'name': model,
                'values': ap_values
            })
            
        return f"""
        <div class="comparison-section">
            <h3>Per-Class Performance (Top Variance Classes)</h3>
            <p>Showing classes with highest performance variance across models</p>
            <div id="class-comparison-chart" class="chart-container"></div>
        </div>
        
        <script>
            (function() {{
                const classNames = {json.dumps(top_class_names)};
                const chartData = {json.dumps(chart_data)};
                
                const traces = chartData.map(data => ({{
                    x: classNames,
                    y: data.values,
                    name: data.name,
                    type: 'bar',
                    text: data.values.map(v => v.toFixed(1) + '%'),
                    textposition: 'outside'
                }}));
                
                const layout = {{
                    barmode: 'group',
                    title: 'AP@0.5 by Class',
                    xaxis: {{
                        title: 'Class',
                        tickangle: -45
                    }},
                    yaxis: {{
                        title: 'AP@0.5 (%)',
                        range: [0, 105]
                    }},
                    height: 500
                }};
                
                Plotly.newPlot('class-comparison-chart', traces, layout, {{responsive: true}});
            }})();
        </script>
        """
        
    def _generate_speed_vs_accuracy_section(self, comparison: Dict) -> str:
        """Generate speed vs accuracy scatter plot."""
        models = comparison['models']
        f1_scores = comparison['metrics']['f1_score']
        inference_times = comparison['metrics']['avg_inference_time']
        
        points_data = []
        for i, model in enumerate(models):
            points_data.append({
                'name': model,
                'x': inference_times[i],
                'y': f1_scores[i] * 100
            })
            
        return f"""
        <div class="comparison-section">
            <h3>Speed vs Accuracy Trade-off</h3>
            <div id="speed-accuracy-chart" class="chart-container"></div>
        </div>
        
        <script>
            (function() {{
                const pointsData = {json.dumps(points_data)};
                
                const trace = {{
                    x: pointsData.map(p => p.x),
                    y: pointsData.map(p => p.y),
                    mode: 'markers+text',
                    type: 'scatter',
                    text: pointsData.map(p => p.name),
                    textposition: 'top center',
                    marker: {{
                        size: 12,
                        color: pointsData.map(p => p.y),
                        colorscale: 'Viridis',
                        showscale: true,
                        colorbar: {{
                            title: 'F1 Score (%)'
                        }}
                    }},
                    hovertemplate: '<b>%{{text}}</b><br>' +
                                  'Inference Time: %{{x:.3f}}s<br>' +
                                  'F1 Score: %{{y:.1f}}%<br>' +
                                  '<extra></extra>'
                }};
                
                const layout = {{
                    title: 'Speed vs Accuracy Trade-off',
                    xaxis: {{
                        title: 'Average Inference Time (seconds)',
                        type: 'log'
                    }},
                    yaxis: {{
                        title: 'F1 Score (%)',
                        range: [0, 105]
                    }},
                    height: 500,
                    annotations: [
                        {{
                            x: Math.min(...pointsData.map(p => p.x)),
                            y: Math.max(...pointsData.map(p => p.y)),
                            text: 'Ideal: Fast & Accurate',
                            showarrow: true,
                            arrowhead: 2,
                            ax: 40,
                            ay: -40
                        }}
                    ]
                }};
                
                Plotly.newPlot('speed-accuracy-chart', [trace], layout, {{responsive: true}});
            }})();
        </script>
        """
        
    def _generate_recommendation_section(self, comparison: Dict) -> str:
        """Generate model recommendations based on use case."""
        models = comparison['models']
        best_models = comparison['best_models']
        
        # Analyze for different use cases
        recommendations = []
        
        # Best overall
        if 'f1_score' in best_models:
            recommendations.append({
                'use_case': 'Best Overall Performance',
                'model': best_models['f1_score'],
                'reason': 'Highest F1 score, balancing precision and recall'
            })
            
        # Best for real-time
        if 'avg_inference_time' in best_models:
            fast_model = best_models['avg_inference_time']
            model_idx = models.index(fast_model)
            f1 = comparison['metrics']['f1_score'][model_idx]
            if f1 > 0.5:  # Reasonable accuracy
                recommendations.append({
                    'use_case': 'Real-time Applications',
                    'model': fast_model,
                    'reason': f'Fastest inference time with acceptable accuracy (F1: {f1:.1%})'
                })
                
        # Best for high precision
        if 'precision' in best_models:
            recommendations.append({
                'use_case': 'High Precision Requirements',
                'model': best_models['precision'],
                'reason': 'Lowest false positive rate, best when false alarms are costly'
            })
            
        # Best for high recall
        if 'recall' in best_models:
            recommendations.append({
                'use_case': 'High Recall Requirements',
                'model': best_models['recall'],
                'reason': 'Lowest false negative rate, best when missing detections is costly'
            })
            
        rec_html = []
        for rec in recommendations:
            rec_html.append(f"""
                <div style="margin: 15px 0; padding: 15px; background: #e8f5e9; border-radius: 5px;">
                    <h4 style="margin: 0 0 10px 0; color: #2e7d32;">{rec['use_case']}</h4>
                    <strong>Recommended Model:</strong> {rec['model']}<br>
                    <strong>Reason:</strong> {rec['reason']}
                </div>
            """)
            
        return f"""
        <div class="comparison-section">
            <h3>Model Recommendations by Use Case</h3>
            {''.join(rec_html)}
        </div>
        """
        
    def export_comparison_data(self, output_path: str):
        """Export comparison data to JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        comparison_data = {
            'overall': self.compare_overall_metrics(),
            'per_class': self.compare_per_class_performance(),
            'models_metadata': {}
        }
        
        # Add metadata for each model
        for model_name, results in self.models_data.items():
            metadata = results.get('metadata', {})
            comparison_data['models_metadata'][model_name] = {
                'model_path': metadata.get('model_path', ''),
                'test_folder': metadata.get('test_folder', ''),
                'conf_threshold': metadata.get('conf_threshold', 0),
                'iou_threshold': metadata.get('iou_threshold', 0),
                'num_images': metadata.get('num_images', 0),
                'timestamp': metadata.get('timestamp', '')
            }
            
        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
            
        print(f"[INFO] Comparison data exported to {output_path}")