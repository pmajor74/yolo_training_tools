"""Advanced benchmark metrics calculator for YOLO model evaluation with COCO-standard metrics."""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import traceback


class AdvancedBenchmarkCalculator:
    """Calculate comprehensive benchmark metrics for object detection."""
    
    def __init__(self, class_names: Dict[int, str], 
                 iou_thresholds: Optional[List[float]] = None,
                 area_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """Initialize advanced calculator with class names and evaluation parameters.
        
        Args:
            class_names: Dictionary mapping class IDs to names
            iou_thresholds: List of IoU thresholds for mAP calculation (default: COCO standard)
            area_ranges: Dictionary of area ranges for size-based analysis
        """
        # Ensure all keys are integers
        self.class_names = {}
        for k, v in class_names.items():
            try:
                self.class_names[int(k)] = v
            except (ValueError, TypeError):
                print(f"[WARNING] AdvancedBenchmarkCalculator: Skipping non-integer class ID: {k}")
                continue
                
        self.num_classes = len(self.class_names)
        
        # COCO-standard IoU thresholds (0.5:0.95 with step 0.05)
        if iou_thresholds is None:
            self.iou_thresholds = [0.5 + i * 0.05 for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds
            
        # COCO-standard area ranges (in pixels squared, normalized here)
        if area_ranges is None:
            self.area_ranges = {
                'small': (0, 0.01),      # < 32^2 pixels in normalized coords
                'medium': (0.01, 0.09),  # 32^2 to 96^2 pixels
                'large': (0.09, 1.0)     # > 96^2 pixels
            }
        else:
            self.area_ranges = area_ranges
            
        # Initialize confusion matrix
        self.confusion_matrix = None
        self.reset_confusion_matrix()
        
        # Store detailed results for advanced analysis
        self.all_detections = []
        self.all_ground_truths = []
        
    def reset_confusion_matrix(self):
        """Reset the confusion matrix."""
        # +1 for background/missed detections
        self.confusion_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1))
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes in normalized center format [x, y, w, h]."""
        # Convert center format to corner format
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
            
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
            
        return inter_area / union_area
        
    def get_box_area_category(self, box: List[float]) -> str:
        """Get the size category of a bounding box."""
        area = box[2] * box[3]  # width * height in normalized coords
        
        for category, (min_area, max_area) in self.area_ranges.items():
            if min_area <= area < max_area:
                return category
        return 'large'  # Default to large if out of bounds
        
    def match_predictions_multi_iou(self, ground_truth: List[Dict], predictions: List[Dict]) -> Dict[float, Dict]:
        """Match predictions to ground truth at multiple IoU thresholds.
        
        Returns:
            Dictionary mapping IoU thresholds to match results
        """
        results = {}
        
        for iou_threshold in self.iou_thresholds:
            matched_gt = set()
            matched_pred = set()
            true_positives = []
            
            # Sort predictions by confidence (highest first)
            sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
            
            for pred_idx, pred in enumerate(sorted_predictions):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(ground_truth):
                    # Check if classes match
                    if pred['class_id'] != gt['class_id']:
                        continue
                        
                    # Check if GT already matched
                    if gt_idx in matched_gt:
                        continue
                        
                    # Calculate IoU
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        
                # If we found a match above threshold
                if best_iou >= iou_threshold:
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                    true_positives.append({
                        'pred_idx': pred_idx,
                        'gt_idx': best_gt_idx,
                        'iou': best_iou,
                        'class_id': pred['class_id'],
                        'confidence': pred['confidence'],
                        'pred_box': pred['bbox'],
                        'gt_box': ground_truth[best_gt_idx]['bbox']
                    })
                    
            # False positives: predictions not matched
            false_positives = [pred for i, pred in enumerate(sorted_predictions) 
                              if i not in matched_pred]
            
            # False negatives: ground truth not matched
            false_negatives = [gt for i, gt in enumerate(ground_truth) 
                              if i not in matched_gt]
            
            results[iou_threshold] = {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
            
        return results
        
    def update_confusion_matrix(self, ground_truth: List[Dict], predictions: List[Dict], iou_threshold: float = 0.5):
        """Update confusion matrix with detections from one image."""
        matches = self.match_predictions_multi_iou(ground_truth, predictions)
        match_result = matches[iou_threshold]
        
        # True positives (might be class confusion)
        for tp in match_result['true_positives']:
            gt_class = tp['class_id']
            # In object detection, TP means correct class, so diagonal
            self.confusion_matrix[gt_class][gt_class] += 1
            
        # False positives (predicted something that wasn't there)
        for fp in match_result['false_positives']:
            pred_class = fp['class_id']
            # Check if it's a misclassification of another object
            best_iou = 0
            best_gt_class = self.num_classes  # Default to background
            
            for gt in ground_truth:
                iou = self.calculate_iou(fp['bbox'], gt['bbox'])
                if iou > best_iou and iou > 0.1:  # Some overlap but wrong class
                    best_iou = iou
                    best_gt_class = gt['class_id']
                    
            self.confusion_matrix[best_gt_class][pred_class] += 1
            
        # False negatives (missed detections)
        for fn in match_result['false_negatives']:
            gt_class = fn['class_id']
            self.confusion_matrix[gt_class][self.num_classes] += 1  # Predicted as background
            
    def calculate_image_metrics_advanced(self, ground_truth: List[Dict], predictions: List[Dict], 
                                        image_path: str = "") -> Dict:
        """Calculate comprehensive metrics for a single image."""
        
        # Store for later analysis
        self.all_detections.extend(predictions)
        self.all_ground_truths.extend(ground_truth)
        
        # Update confusion matrix
        self.update_confusion_matrix(ground_truth, predictions)
        
        # Get matches at all IoU thresholds
        all_matches = self.match_predictions_multi_iou(ground_truth, predictions)
        
        # Calculate metrics at each IoU threshold
        metrics_by_iou = {}
        for iou_thresh, matches in all_matches.items():
            tp_count = len(matches['true_positives'])
            fp_count = len(matches['false_positives'])
            fn_count = len(matches['false_negatives'])
            
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_by_iou[iou_thresh] = {
                'true_positives': tp_count,
                'false_positives': fp_count,
                'false_negatives': fn_count,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp_details': matches['true_positives'],
                'fp_details': matches['false_positives'],
                'fn_details': matches['false_negatives']
            }
            
        # Size-based analysis at IoU 0.5
        size_metrics = self.calculate_size_based_metrics(
            ground_truth, predictions, all_matches[0.5]
        )
        
        # Error categorization
        error_analysis = self.analyze_errors(all_matches[0.5])
        
        # Confidence statistics
        confidence_stats = self.calculate_confidence_statistics(predictions, all_matches[0.5])
        
        return {
            'image_path': image_path,
            'ground_truth_count': len(ground_truth),
            'prediction_count': len(predictions),
            'metrics_by_iou': metrics_by_iou,
            'size_metrics': size_metrics,
            'error_analysis': error_analysis,
            'confidence_stats': confidence_stats,
            'ground_truth_details': ground_truth,
            'prediction_details': predictions
        }
        
    def calculate_size_based_metrics(self, ground_truth: List[Dict], predictions: List[Dict], 
                                    matches: Dict) -> Dict:
        """Calculate metrics based on object sizes."""
        size_stats = {category: {'tp': 0, 'fp': 0, 'fn': 0} 
                     for category in self.area_ranges.keys()}
        
        # Count TPs by size
        for tp in matches['true_positives']:
            size_cat = self.get_box_area_category(tp['gt_box'])
            size_stats[size_cat]['tp'] += 1
            
        # Count FPs by size
        for fp in matches['false_positives']:
            size_cat = self.get_box_area_category(fp['bbox'])
            size_stats[size_cat]['fp'] += 1
            
        # Count FNs by size
        for fn in matches['false_negatives']:
            size_cat = self.get_box_area_category(fn['bbox'])
            size_stats[size_cat]['fn'] += 1
            
        # Calculate precision/recall per size
        for category in size_stats:
            stats = size_stats[category]
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            
            stats['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            stats['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            stats['f1_score'] = (2 * stats['precision'] * stats['recall'] / 
                               (stats['precision'] + stats['recall']) 
                               if (stats['precision'] + stats['recall']) > 0 else 0)
            
        return size_stats
        
    def analyze_errors(self, matches: Dict) -> Dict:
        """Categorize errors into different types."""
        error_types = {
            'localization_errors': [],  # Correct class but poor IoU
            'classification_errors': [],  # Wrong class prediction
            'background_fps': [],  # False positives on background
            'duplicate_detections': [],  # Multiple detections for same object
            'missed_detections': []  # Completely missed objects
        }
        
        # Analyze TPs for localization quality
        for tp in matches['true_positives']:
            if tp['iou'] < 0.75:
                error_types['localization_errors'].append({
                    'class_id': tp['class_id'],
                    'iou': tp['iou'],
                    'confidence': tp['confidence']
                })
                
        # Analyze FPs
        for fp in matches['false_positives']:
            # Check if this FP overlaps significantly with any GT
            overlaps_with_gt = False
            for gt in self.all_ground_truths[-len(matches['false_negatives'])-len(matches['true_positives']):]:
                iou = self.calculate_iou(fp['bbox'], gt['bbox'])
                if iou > 0.1:
                    overlaps_with_gt = True
                    if gt['class_id'] != fp['class_id']:
                        error_types['classification_errors'].append({
                            'predicted_class': fp['class_id'],
                            'actual_class': gt['class_id'],
                            'confidence': fp['confidence'],
                            'iou': iou
                        })
                    break
                    
            if not overlaps_with_gt:
                error_types['background_fps'].append({
                    'class_id': fp['class_id'],
                    'confidence': fp['confidence']
                })
                
        # FNs are missed detections
        error_types['missed_detections'] = [{
            'class_id': fn['class_id'],
            'size': self.get_box_area_category(fn['bbox'])
        } for fn in matches['false_negatives']]
        
        return error_types
        
    def calculate_confidence_statistics(self, predictions: List[Dict], matches: Dict) -> Dict:
        """Calculate confidence distribution statistics."""
        tp_confidences = [tp['confidence'] for tp in matches['true_positives']]
        fp_confidences = [fp['confidence'] for fp in matches['false_positives']]
        
        all_confidences = [p['confidence'] for p in predictions]
        
        return {
            'tp_confidence_mean': np.mean(tp_confidences) if tp_confidences else 0,
            'tp_confidence_std': np.std(tp_confidences) if tp_confidences else 0,
            'tp_confidence_min': min(tp_confidences) if tp_confidences else 0,
            'tp_confidence_max': max(tp_confidences) if tp_confidences else 0,
            'fp_confidence_mean': np.mean(fp_confidences) if fp_confidences else 0,
            'fp_confidence_std': np.std(fp_confidences) if fp_confidences else 0,
            'fp_confidence_min': min(fp_confidences) if fp_confidences else 0,
            'fp_confidence_max': max(fp_confidences) if fp_confidences else 0,
            'all_confidence_mean': np.mean(all_confidences) if all_confidences else 0,
            'all_confidence_histogram': np.histogram(all_confidences, bins=10, range=(0, 1))[0].tolist() if all_confidences else []
        }
        
    def calculate_ap_at_iou(self, class_results: List[Dict], num_gt: int, iou_threshold: float) -> float:
        """Calculate Average Precision for a single class at specific IoU threshold."""
        if num_gt == 0:
            return 0.0
            
        # Sort by confidence
        sorted_results = sorted(class_results, key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(sorted_results))
        fp = np.zeros(len(sorted_results))
        
        for i, result in enumerate(sorted_results):
            if result['is_tp_at_iou'].get(iou_threshold, False):
                tp[i] = 1
            else:
                fp[i] = 1
                
        # Cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / num_gt
        
        # Add boundary points
        precision = np.concatenate([[1], precision])
        recall = np.concatenate([[0], recall])
        
        # Calculate AP using interpolation
        # Use all unique recall values as thresholds
        mrec = recall
        mpre = precision
        
        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            
        # Calculate area under PR curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return ap
        
    def calculate_overall_metrics_advanced(self, image_results: List[Dict]) -> Dict:
        """Calculate comprehensive overall metrics across all images."""
        try:
            print("[INFO] AdvancedBenchmarkCalculator.calculate_overall_metrics_advanced: Starting calculation")
            
            # Initialize aggregators
            total_by_iou = {iou: {'tp': 0, 'fp': 0, 'fn': 0} for iou in self.iou_thresholds}
            per_class_data = defaultdict(lambda: {
                'predictions': [],
                'gt_count': 0,
                'by_iou': {iou: {'tp': 0, 'fp': 0, 'fn': 0} for iou in self.iou_thresholds}
            })
            
            # Aggregate results across images
            for img_result in image_results:
                metrics_by_iou = img_result.get('metrics_by_iou', {})
                
                # Use metrics at IoU 0.5 as reference for collecting detections
                # This avoids duplicating predictions and ground truths
                reference_iou = 0.5
                if reference_iou in metrics_by_iou:
                    ref_metrics = metrics_by_iou[reference_iou]
                    
                    # Store detections once (for AP calculation)
                    for tp in ref_metrics['tp_details']:
                        class_id = tp['class_id']
                        
                        # Store detection for AP calculation
                        detection = {
                            'confidence': tp['confidence'],
                            'is_tp_at_iou': {}
                        }
                        for check_iou in self.iou_thresholds:
                            detection['is_tp_at_iou'][check_iou] = tp['iou'] >= check_iou
                        per_class_data[class_id]['predictions'].append(detection)
                        
                        # Count as ground truth (only once)
                        per_class_data[class_id]['gt_count'] += 1
                        
                    for fp in ref_metrics['fp_details']:
                        class_id = fp['class_id']
                        
                        # Store as FP for AP calculation
                        detection = {
                            'confidence': fp['confidence'],
                            'is_tp_at_iou': {iou: False for iou in self.iou_thresholds}
                        }
                        per_class_data[class_id]['predictions'].append(detection)
                        
                    for fn in ref_metrics['fn_details']:
                        class_id = fn['class_id']
                        # Count as ground truth (missed detection)
                        per_class_data[class_id]['gt_count'] += 1
                
                # Now accumulate metrics for each IoU threshold
                for iou_thresh, metrics in metrics_by_iou.items():
                    total_by_iou[iou_thresh]['tp'] += metrics['true_positives']
                    total_by_iou[iou_thresh]['fp'] += metrics['false_positives']
                    total_by_iou[iou_thresh]['fn'] += metrics['false_negatives']
                    
                    # Update per-class counts for this IoU
                    for tp in metrics['tp_details']:
                        class_id = tp['class_id']
                        per_class_data[class_id]['by_iou'][iou_thresh]['tp'] += 1
                        
                    for fp in metrics['fp_details']:
                        class_id = fp['class_id']
                        per_class_data[class_id]['by_iou'][iou_thresh]['fp'] += 1
                        
                    for fn in metrics['fn_details']:
                        class_id = fn['class_id']
                        per_class_data[class_id]['by_iou'][iou_thresh]['fn'] += 1
                        
            # Calculate overall metrics at each IoU
            overall_metrics_by_iou = {}
            for iou_thresh in self.iou_thresholds:
                tp = total_by_iou[iou_thresh]['tp']
                fp = total_by_iou[iou_thresh]['fp']
                fn = total_by_iou[iou_thresh]['fn']
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                overall_metrics_by_iou[iou_thresh] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn
                }
                
            # Calculate per-class AP at different IoU thresholds
            per_class_metrics = {}
            map_values_by_iou = {iou: [] for iou in self.iou_thresholds}
            
            for class_id, class_data in per_class_data.items():
                class_name = self.class_names.get(class_id, f"Class_{class_id}")
                per_class_metrics[class_name] = {
                    'ap_by_iou': {},
                    'metrics_by_iou': {}
                }
                
                # Calculate AP at each IoU threshold
                for iou_thresh in self.iou_thresholds:
                    ap = self.calculate_ap_at_iou(
                        class_data['predictions'],
                        class_data['gt_count'],
                        iou_thresh
                    )
                    per_class_metrics[class_name]['ap_by_iou'][iou_thresh] = ap
                    
                    if class_data['gt_count'] > 0:  # Only include in mAP if class exists
                        map_values_by_iou[iou_thresh].append(ap)
                        
                    # Also store precision/recall at this IoU
                    tp = class_data['by_iou'][iou_thresh]['tp']
                    fp = class_data['by_iou'][iou_thresh]['fp']
                    fn = class_data['by_iou'][iou_thresh]['fn']
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    per_class_metrics[class_name]['metrics_by_iou'][iou_thresh] = {
                        'precision': precision,
                        'recall': recall,
                        'true_positives': tp,
                        'false_positives': fp,
                        'false_negatives': fn
                    }
                    
            # Calculate mAP values
            map_by_iou = {}
            for iou_thresh in self.iou_thresholds:
                if map_values_by_iou[iou_thresh]:
                    map_by_iou[iou_thresh] = np.mean(map_values_by_iou[iou_thresh])
                else:
                    map_by_iou[iou_thresh] = 0.0
                    
            # COCO-style mAP@[0.5:0.95]
            map_50_95 = np.mean(list(map_by_iou.values()))
            
            # Common specific values
            map_50 = map_by_iou.get(0.5, 0.0)
            map_75 = map_by_iou.get(0.75, 0.0)
            
            # Get metrics at IoU 0.5 for backward compatibility
            metrics_at_50 = overall_metrics_by_iou[0.5]
            
            print(f"[INFO] AdvancedBenchmarkCalculator.calculate_overall_metrics_advanced: Calculated mAP@0.5={map_50:.4f}, mAP@0.75={map_75:.4f}, mAP@[0.5:0.95]={map_50_95:.4f}")
            
            return {
                # Primary COCO metrics
                'map_50': map_50,
                'map_75': map_75,
                'map_50_95': map_50_95,
                'map_by_iou': map_by_iou,
                
                # Overall metrics at IoU 0.5 (for compatibility)
                'precision': metrics_at_50['precision'],
                'recall': metrics_at_50['recall'],
                'f1_score': metrics_at_50['f1_score'],
                'true_positives': metrics_at_50['true_positives'],
                'false_positives': metrics_at_50['false_positives'],
                'false_negatives': metrics_at_50['false_negatives'],
                
                # Detailed metrics
                'overall_metrics_by_iou': overall_metrics_by_iou,
                'per_class_metrics': per_class_metrics,
                
                # Confusion matrix
                'confusion_matrix': self.confusion_matrix.tolist(),
                
                # Additional statistics
                'total_images': len(image_results),
                'total_ground_truth': sum(r['ground_truth_count'] for r in image_results),
                'total_predictions': sum(r['prediction_count'] for r in image_results)
            }
            
        except Exception as e:
            print(f"[ERROR] AdvancedBenchmarkCalculator.calculate_overall_metrics_advanced: {str(e)}")
            traceback.print_exc()
            raise
            
    def calculate_precision_recall_curve(self, class_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate precision-recall curve for a specific class or overall."""
        # This would need to aggregate all predictions across images
        # For now, returning placeholder
        recall = np.linspace(0, 1, 100)
        precision = np.ones_like(recall) * 0.8 - recall * 0.3  # Placeholder
        return precision, recall
        
    def get_optimal_threshold(self, target_metric: str = 'f1') -> float:
        """Find optimal confidence threshold for given metric."""
        # Would iterate through different thresholds and find best
        # For now, returning default
        return 0.25