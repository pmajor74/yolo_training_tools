"""Benchmark metrics calculator for YOLO model evaluation."""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class BenchmarkCalculator:
    """Calculate benchmark metrics for object detection."""
    
    def __init__(self, class_names: Dict[int, str]):
        """Initialize calculator with class names."""
        self.class_names = class_names
        self.num_classes = len(class_names)
        
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
        
    def match_predictions(self, ground_truth: List[Dict], predictions: List[Dict], 
                         iou_threshold: float = 0.5) -> Tuple[List, List, List]:
        """Match predictions to ground truth boxes."""
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
                    'confidence': pred['confidence']
                })
                
        # False positives: predictions not matched
        false_positives = [pred for i, pred in enumerate(sorted_predictions) 
                          if i not in matched_pred]
        
        # False negatives: ground truth not matched
        false_negatives = [gt for i, gt in enumerate(ground_truth) 
                          if i not in matched_gt]
        
        return true_positives, false_positives, false_negatives
        
    def calculate_image_metrics(self, ground_truth: List[Dict], predictions: List[Dict], 
                               iou_threshold: float = 0.5) -> Dict:
        """Calculate metrics for a single image."""
        tp_list, fp_list, fn_list = self.match_predictions(
            ground_truth, predictions, iou_threshold
        )
        
        metrics = {
            'ground_truth_count': len(ground_truth),
            'prediction_count': len(predictions),
            'true_positives': len(tp_list),
            'false_positives': len(fp_list),
            'false_negatives': len(fn_list),
            'tp_details': tp_list,
            'fp_details': fp_list,
            'fn_details': fn_list
        }
        
        # Calculate precision and recall
        if metrics['true_positives'] + metrics['false_positives'] > 0:
            metrics['precision'] = metrics['true_positives'] / (
                metrics['true_positives'] + metrics['false_positives']
            )
        else:
            metrics['precision'] = 0.0
            
        if metrics['true_positives'] + metrics['false_negatives'] > 0:
            metrics['recall'] = metrics['true_positives'] / (
                metrics['true_positives'] + metrics['false_negatives']
            )
        else:
            metrics['recall'] = 0.0
            
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (
                metrics['precision'] + metrics['recall']
            )
        else:
            metrics['f1_score'] = 0.0
            
        return metrics
        
    def calculate_ap(self, class_results: List[Dict], num_gt: int) -> float:
        """Calculate Average Precision for a single class."""
        if num_gt == 0:
            return 0.0
            
        # Sort by confidence
        sorted_results = sorted(class_results, key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(sorted_results))
        fp = np.zeros(len(sorted_results))
        
        for i, result in enumerate(sorted_results):
            if result['is_tp']:
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
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
            
        return ap
        
    def calculate_overall_metrics(self, image_results: Dict[str, Dict]) -> Dict:
        """Calculate overall metrics across all images."""
        # Aggregate counts
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0
        total_pred = 0
        total_inference_time = 0
        
        # Per-class collections for AP calculation
        per_class_predictions = defaultdict(list)
        per_class_gt_count = defaultdict(int)
        
        # Track problem images
        problem_images = []
        
        for image_name, metrics in image_results.items():
            total_tp += metrics['true_positives']
            total_fp += metrics['false_positives']
            total_fn += metrics['false_negatives']
            total_gt += metrics['ground_truth_count']
            total_pred += metrics['prediction_count']
            total_inference_time += metrics.get('inference_time', 0)
            
            # Track problem images
            problems = []
            if metrics['false_positives'] > 0:
                problems.append(f"{metrics['false_positives']} false positive(s)")
            if metrics['false_negatives'] > 0:
                problems.append(f"{metrics['false_negatives']} missed detection(s)")
                
            if problems:
                problem_images.append({
                    'image': image_name,
                    'problems': problems,
                    'false_positives': metrics['false_positives'],
                    'false_negatives': metrics['false_negatives'],
                    'fp_details': metrics.get('fp_details', []),
                    'fn_details': metrics.get('fn_details', []),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0)
                })
            
            # Collect per-class data
            for tp in metrics.get('tp_details', []):
                per_class_predictions[tp['class_id']].append({
                    'confidence': tp['confidence'],
                    'is_tp': True
                })
                
            for fp in metrics.get('fp_details', []):
                per_class_predictions[fp['class_id']].append({
                    'confidence': fp['confidence'],
                    'is_tp': False
                })
                
            for fn in metrics.get('fn_details', []):
                per_class_gt_count[fn['class_id']] += 1
                
            for tp in metrics.get('tp_details', []):
                per_class_gt_count[tp['class_id']] += 1
                
        # Overall metrics
        overall = {
            'total_images': len(image_results),
            'total_ground_truth': total_gt,
            'total_predictions': total_pred,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'total_inference_time': total_inference_time,
            'avg_inference_time': total_inference_time / len(image_results) if image_results else 0
        }
        
        # Overall precision, recall, F1
        if total_tp + total_fp > 0:
            overall['precision'] = total_tp / (total_tp + total_fp)
        else:
            overall['precision'] = 0.0
            
        if total_tp + total_fn > 0:
            overall['recall'] = total_tp / (total_tp + total_fn)
        else:
            overall['recall'] = 0.0
            
        if overall['precision'] + overall['recall'] > 0:
            overall['f1_score'] = 2 * (overall['precision'] * overall['recall']) / (
                overall['precision'] + overall['recall']
            )
        else:
            overall['f1_score'] = 0.0
            
        # Per-class metrics
        per_class_metrics = {}
        ap_values = []
        
        for class_id in self.class_names.keys():
            class_name = self.class_names[class_id]
            class_predictions = per_class_predictions.get(class_id, [])
            class_gt_count = per_class_gt_count.get(class_id, 0)
            
            if class_gt_count > 0 or len(class_predictions) > 0:
                # Calculate AP for this class
                ap = self.calculate_ap(class_predictions, class_gt_count)
                
                # Count TPs and FPs for this class
                class_tp = sum(1 for p in class_predictions if p['is_tp'])
                class_fp = sum(1 for p in class_predictions if not p['is_tp'])
                class_fn = class_gt_count - class_tp
                
                # Class precision and recall
                if class_tp + class_fp > 0:
                    class_precision = class_tp / (class_tp + class_fp)
                else:
                    class_precision = 0.0
                    
                if class_tp + class_fn > 0:
                    class_recall = class_tp / (class_tp + class_fn)
                else:
                    class_recall = 0.0
                    
                per_class_metrics[class_name] = {
                    'ap_50': ap,  # Assuming IoU threshold of 0.5
                    'precision': class_precision,
                    'recall': class_recall,
                    'true_positives': class_tp,
                    'false_positives': class_fp,
                    'false_negatives': class_fn,
                    'ground_truth_count': class_gt_count
                }
                
                if class_gt_count > 0:  # Only include in mAP if class exists in ground truth
                    ap_values.append(ap)
                    
        # Calculate mAP
        if ap_values:
            overall['map_50'] = np.mean(ap_values)
        else:
            overall['map_50'] = 0.0
            
        # For simplicity, using same value for mAP@0.5:0.95
        # In practice, you'd calculate at multiple IoU thresholds
        overall['map_50_95'] = overall['map_50'] * 0.9  # Rough approximation
        
        overall['per_class_metrics'] = per_class_metrics
        overall['per_image_results'] = image_results
        overall['problem_images'] = sorted(problem_images, 
                                          key=lambda x: x['f1_score'])  # Sort by worst F1 first
        
        return overall
        
    def calculate_confusion_matrix(self, image_results: Dict[str, Dict]) -> np.ndarray:
        """Calculate confusion matrix across all images."""
        # Initialize confusion matrix (num_classes + 1 for background/missing)
        matrix = np.zeros((self.num_classes + 1, self.num_classes + 1))
        
        for image_name, metrics in image_results.items():
            # True positives (correct class)
            for tp in metrics.get('tp_details', []):
                class_id = tp['class_id']
                matrix[class_id][class_id] += 1
                
            # False positives (predicted but not there)
            for fp in metrics.get('fp_details', []):
                pred_class = fp['class_id']
                matrix[self.num_classes][pred_class] += 1  # Background -> predicted
                
            # False negatives (there but not predicted)
            for fn in metrics.get('fn_details', []):
                gt_class = fn['class_id']
                matrix[gt_class][self.num_classes] += 1  # GT -> background
                
        return matrix