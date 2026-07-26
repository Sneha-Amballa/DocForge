import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from src.logger import get_logger

logger = get_logger("DocForge.EvalMetrics")

try:
    import sklearn.metrics as skm
    SKLEARN_AVAILABLE = True
    logger.info("Scikit-learn detected. Using sklearn for classification metrics.")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not found. Using custom pure-numpy classification fallback metrics.")

def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[Union[List[float], np.ndarray]] = None
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics.

    Calculates: Accuracy, Precision, Recall, F1-score, Balanced Accuracy,
    ROC-AUC, and Confusion Matrix.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        Dict[str, Any]: Metric names to values mapping.
    """
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    
    if len(y_true) == 0:
        logger.warning("Empty labels provided to compute_classification_metrics.")
        return {}

    if SKLEARN_AVAILABLE:
        try:
            # Confusion Matrix: TN, FP, FN, TP
            tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            metrics = {
                "accuracy": float(skm.accuracy_score(y_true, y_pred)),
                "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(skm.f1_score(y_true, y_pred, zero_division=0)),
                "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp)
                }
            }
            
            # ROC AUC if probabilities are provided and there is at least one sample of each class
            if y_prob is not None and len(np.unique(y_true)) > 1:
                metrics["roc_auc"] = float(skm.roc_auc_score(y_true, y_prob))
            else:
                metrics["roc_auc"] = 0.5
                
            return metrics
        except Exception as e:
            logger.error(f"Error computing metrics with sklearn: {e}. Falling back to NumPy.")

    # NumPy/Native python fallback
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Balanced Accuracy: average of recall on each class
    rec_pos = recall
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (rec_pos + rec_neg) / 2.0
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        },
        "roc_auc": 0.5  # default baseline for custom numpy ROC-AUC
    }
    
    # Custom simple ROC-AUC estimation if probabilities provided
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            # Sort by probability
            y_prob = np.array(y_prob, dtype=np.float32)
            desc_score_indices = np.argsort(y_prob)[::-1]
            y_true_sorted = y_true[desc_score_indices]
            
            # Count cumulative sum of TP and FP
            tps = np.cumsum(y_true_sorted == 1)
            fps = np.cumsum(y_true_sorted == 0)
            
            # Total counts
            n_pos = tps[-1]
            n_neg = fps[-1]
            
            if n_pos > 0 and n_neg > 0:
                # Integrate trapezoidal area under ROC curve
                tps = np.r_[0, tps]
                fps = np.r_[0, fps]
                
                # Normalize to [0.0, 1.0]
                tpr = tps / n_pos
                fpr = fps / n_neg
                
                auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0)
                metrics["roc_auc"] = float(auc)
        except Exception:
            pass

    return metrics

def save_metrics_to_json_csv(
    metrics: Dict[str, Any],
    save_dir: Union[str, Path],
    filename_prefix: str = "evaluation_summary"
) -> None:
    """Save metrics dictionary to JSON and CSV formats.

    Args:
        metrics: Dictionary of calculated performance metrics.
        save_dir: Directory where files should be written.
        filename_prefix: Base name of output files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = save_dir / f"{filename_prefix}.json"
    csv_path = save_dir / f"{filename_prefix}.csv"
    
    # 1. Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved classification metrics in JSON: {json_path}")
    
    # 2. Save CSV
    # Flatten confusion matrix dictionary
    flat_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_metrics[f"{k}_{sub_k}"] = sub_v
        else:
            flat_metrics[k] = v
            
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Metric,Value\n")
        for k, v in flat_metrics.items():
            f.write(f"{k},{v}\n")
    logger.info(f"Saved classification metrics in CSV: {csv_path}")
