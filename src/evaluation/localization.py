from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.logger import get_logger

logger = get_logger("DocForge.EvalLocalization")

def compute_bbox_iou(boxA: List[Union[int, float]], boxB: List[Union[int, float]]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        boxA: Bounding box coordinates [xmin, ymin, xmax, ymax].
        boxB: Bounding box coordinates [xmin, ymin, xmax, ymax].

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    # Determine intersection corners
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    # Compute individual box areas
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute union area
    union_area = areaA + areaB - inter_area

    if union_area <= 0:
        return 0.0
        
    return float(inter_area / union_area)

def evaluate_localization_set(
    gt_boxes_list: List[List[List[int]]],
    pred_boxes_list: List[List[List[int]]],
    iou_thresholds: List[float] = [0.3, 0.5, 0.7]
) -> Dict[str, Any]:
    """Compute dataset-wide localization performance (mAP, average IoU, recalls).

    Args:
        gt_boxes_list: Nested list of ground truth bounding boxes per sample.
        pred_boxes_list: Nested list of predicted bounding boxes per sample.
        iou_thresholds: IoU cutoff thresholds for evaluation.

    Returns:
        Dict[str, Any]: Compiled localization metrics.
    """
    total_samples = len(gt_boxes_list)
    if total_samples == 0:
        return {}

    ious = []
    
    # Store performance at thresholds
    tp_at_thresholds = {t: 0 for t in iou_thresholds}
    fp_at_thresholds = {t: 0 for t in iou_thresholds}
    fn_at_thresholds = {t: 0 for t in iou_thresholds}

    for gt_boxes, pred_boxes in zip(gt_boxes_list, pred_boxes_list):
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            # Correct authentic prediction
            for t in iou_thresholds:
                pass
            continue
            
        if len(gt_boxes) == 0 and len(pred_boxes) > 0:
            # False alarms (all predicted boxes are FP)
            for t in iou_thresholds:
                fp_at_thresholds[t] += len(pred_boxes)
            ious.append(0.0)
            continue
            
        if len(gt_boxes) > 0 and len(pred_boxes) == 0:
            # Misses (all ground truth boxes are FN)
            for t in iou_thresholds:
                fn_at_thresholds[t] += len(gt_boxes)
            ious.append(0.0)
            continue

        # Both have boxes: find matched pairs using greedy IoU pairing
        matched_gt = set()
        sample_ious = []
        
        for p_box in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, g_box in enumerate(gt_boxes):
                iou = compute_bbox_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
            sample_ious.append(best_iou)
            
            # Record metrics at thresholds
            for t in iou_thresholds:
                if best_iou >= t:
                    if best_gt_idx not in matched_gt:
                        tp_at_thresholds[t] += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        # Double prediction of same gt box counts as FP
                        fp_at_thresholds[t] += 1
                else:
                    fp_at_thresholds[t] += 1
                    
        # Any unmatched GT counts as FN
        for t in iou_thresholds:
            unmatched_count = len(gt_boxes) - len(matched_gt)
            fn_at_thresholds[t] += max(0, unmatched_count)
            
        if sample_ious:
            ious.append(float(np.mean(sample_ious)))
        else:
            ious.append(0.0)

    # Calculate average metrics
    avg_iou = float(np.mean(ious)) if ious else 0.0
    
    # Calculate Precision & Recall @ IoU
    precision_at_iou = {}
    recall_at_iou = {}
    f1_at_iou = {}
    
    for t in iou_thresholds:
        tp = tp_at_thresholds[t]
        fp = fp_at_thresholds[t]
        fn = fn_at_thresholds[t]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_at_iou[f"iou_{t}"] = float(precision)
        recall_at_iou[f"iou_{t}"] = float(recall)
        f1_at_iou[f"iou_{t}"] = float(f1)

    # mAP is the mean of F1-scores or Precisions across thresholds
    mAP = float(np.mean(list(precision_at_iou.values())))

    return {
        "average_iou": avg_iou,
        "mAP": mAP,
        "precision_at_iou": precision_at_iou,
        "recall_at_iou": recall_at_iou,
        "f1_at_iou": f1_at_iou
    }

def save_localization_overlay(
    image: Image.Image,
    gt_boxes: List[List[int]],
    pred_boxes: List[List[int]],
    iou_score: float,
    save_path: Union[str, Path]
) -> None:
    """Draw and save visual comparison overlays of ground-truth vs. predicted boxes.

    Draws:
        - Ground truth boxes in GREEN.
        - Predicted boxes in RED.
        - Adds IoU label text.

    Args:
        image: Pillow Image.
        gt_boxes: List of ground-truth absolute boxes.
        pred_boxes: List of predicted absolute boxes.
        iou_score: Matching IoU score.
        save_path: Output PNG filepath.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert image to RGB if not already
    draw_img = image.convert("RGB")
    draw = ImageDraw.Draw(draw_img)
    
    # Draw GT boxes in green
    for box in gt_boxes:
        # width = 3
        draw.rectangle(box, outline="green", width=3)
        draw.text((box[0] + 5, box[1] + 5), "GT", fill="green")

    # Draw Pred boxes in red
    for box in pred_boxes:
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 5, box[3] - 15), f"Pred", fill="red")

    # Add text overlay showing aggregate IoU
    draw.rectangle([10, 10, 150, 40], fill="black")
    draw.text((15, 15), f"IoU score: {iou_score:.3f}", fill="white")

    draw_img.save(save_path)
    logger.debug(f"Saved localization overlay visualization in {save_path}")
