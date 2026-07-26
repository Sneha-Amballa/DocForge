import csv
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np

from src.logger import get_logger

logger = get_logger("DocForge.EvalFailureAnalysis")

def analyze_failures(
    samples_metadata: List[Dict[str, Any]],
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
    gt_boxes_list: List[List[List[int]]],
    pred_boxes_list: List[List[List[int]]],
    save_path: Union[str, Path],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.65
) -> Dict[str, Any]:
    """Isolate, categorize, and log model prediction and localization errors.

    Categories:
        - False Positives (FP)
        - False Negatives (FN)
        - Localization Failures (correct label, but IoU < threshold)
        - Low Confidence Predictions (correct prediction, but low confidence score)

    Args:
        samples_metadata: List of dicts with 'sample_id', 'forgery_type', etc.
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted confidence probabilities.
        gt_boxes_list: Nested list of ground truth bounding boxes.
        pred_boxes_list: Nested list of predicted bounding boxes.
        save_path: Output CSV filepath.
        iou_threshold: Overlap threshold for box accuracy checks.
        confidence_threshold: Confidence boundary cutoff.

    Returns:
        Dict[str, Any]: Error category counts and clustering stats.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    
    failures = []
    
    fp_count = 0
    fn_count = 0
    loc_fail_count = 0
    low_conf_count = 0
    
    # Track statistics grouped by forgery type
    forgery_stats = {}

    from src.evaluation.localization import compute_bbox_iou

    for idx in range(len(y_true)):
        meta = samples_metadata[idx]
        sample_id = meta.get("sample_id", f"sample_{idx}")
        forgery_type = meta.get("forgery_type", "Unknown")
        
        gt = int(y_true[idx])
        pred = int(y_pred[idx])
        prob = float(y_prob[idx])
        conf = prob if pred == 1 else 1.0 - prob
        
        gt_boxes = gt_boxes_list[idx]
        pred_boxes = pred_boxes_list[idx]
        
        failure_category = None
        explanation = meta.get("explanation", "")
        
        # 1. Check classification labels
        if gt == 0 and pred == 1:
            failure_category = "False Positive"
            fp_count += 1
        elif gt == 1 and pred == 0:
            failure_category = "False Negative"
            fn_count += 1
        # 2. Check localization box matches
        elif gt == 1 and pred == 1:
            # Both reported tampered: check box IoU
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                best_iou = 0.0
                for p_box in pred_boxes:
                    for g_box in gt_boxes:
                        iou = compute_bbox_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                if best_iou < iou_threshold:
                    failure_category = "Localization Failure"
                    loc_fail_count += 1
            elif len(gt_boxes) > 0 and len(pred_boxes) == 0:
                failure_category = "Localization Failure"
                loc_fail_count += 1
        
        # 3. Check low confidence on correct predictions
        if failure_category is None and gt == pred and conf < confidence_threshold:
            failure_category = "Low Confidence Prediction"
            low_conf_count += 1
            
        if failure_category is not None:
            # Record error record
            failures.append({
                "sample_id": sample_id,
                "ground_truth": gt,
                "prediction": pred,
                "confidence": f"{conf:.4f}",
                "failure_category": failure_category,
                "forgery_type": forgery_type,
                "num_gt_boxes": len(gt_boxes),
                "num_pred_boxes": len(pred_boxes),
                "explanation": explanation
            })
            
            # Increment forgery type errors count
            forgery_stats[forgery_type] = forgery_stats.get(forgery_type, 0) + 1

    # Write failure cases to CSV
    headers = [
        "sample_id", "ground_truth", "prediction", "confidence",
        "failure_category", "forgery_type", "num_gt_boxes", "num_pred_boxes", "explanation"
    ]
    
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for fail in failures:
            writer.writerow(fail)
            
    logger.info(f"Saved failure analysis database to {save_path} (Total failures: {len(failures)})")
    
    return {
        "total_failures": len(failures),
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "localization_failures": loc_fail_count,
        "low_confidence_correct": low_conf_count,
        "failures_by_forgery_type": forgery_stats
    }
