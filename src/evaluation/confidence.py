import numpy as np
from typing import Dict, Any, List, Union, Tuple

from src.logger import get_logger

logger = get_logger("DocForge.EvalConfidence")

def compute_calibration_error(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    num_bins: int = 10
) -> Dict[str, Any]:
    """Calculate the Expected Calibration Error (ECE) and bin statistics.

    ECE evaluates how well the model's confidence corresponds to true accuracy.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted confidence probabilities (between 0.0 and 1.0).
        num_bins: Number of confidence bins (defaults to 10).

    Returns:
        Dict[str, Any]: Calculated ECE score and list of bin accuracies/confidences.
    """
    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    
    # Clip probabilities to [0.0, 1.0]
    y_prob = np.clip(y_prob, 0.0, 1.0)
    
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    
    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_sizes = []
    
    total_samples = len(y_true)
    if total_samples == 0:
        return {"ece": 0.0, "bins": []}

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select samples within current bin range
        # For the last bin, include the upper boundary
        if i == num_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_size = np.sum(in_bin)
        bin_sizes.append(int(bin_size))
        
        if bin_size > 0:
            # Accuracy is fraction of positive predictions matched
            # Since predictions are binary, we check if:
            # y_prob >= 0.5 is class 1, otherwise class 0
            # Let's map predicted class
            y_pred_bin = (y_prob[in_bin] >= 0.5).astype(np.int32)
            bin_acc = np.mean(y_pred_bin == y_true[in_bin])
            
            # Confidence is average probability assigned
            # For class 0 predictions (prob < 0.5), confidence is (1 - prob)
            # For class 1 predictions (prob >= 0.5), confidence is prob
            conf_adjusted = np.where(y_prob[in_bin] >= 0.5, y_prob[in_bin], 1.0 - y_prob[in_bin])
            bin_conf = np.mean(conf_adjusted)
            
            # Weight ECE by bin occupancy
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
            
            bin_accuracies.append(float(bin_acc))
            bin_confidences.append(float(bin_conf))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    logger.info(f"Expected Calibration Error (ECE): {ece:.5f}")
    
    bins_list = []
    for i in range(num_bins):
        bins_list.append({
            "bin_idx": i,
            "range": [float(bin_boundaries[i]), float(bin_boundaries[i + 1])],
            "size": bin_sizes[i],
            "accuracy": bin_accuracies[i],
            "confidence": bin_confidences[i]
        })
        
    return {
        "ece": float(ece),
        "bins": bins_list
    }

def analyze_confidence_distributions(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray]
) -> Dict[str, Any]:
    """Separate and profile confidence scores for correct vs. incorrect predictions.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted confidence probabilities.

    Returns:
        Dict[str, Any]: Descriptive stats of distributions.
    """
    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    
    # Clip
    y_prob = np.clip(y_prob, 0.0, 1.0)
    
    y_pred = (y_prob >= 0.5).astype(np.int32)
    correct_mask = (y_pred == y_true)
    
    # Calculate confidence values (adjusted for predicted class)
    conf = np.where(y_prob >= 0.5, y_prob, 1.0 - y_prob)
    
    correct_conf = conf[correct_mask]
    incorrect_conf = conf[~correct_mask]
    
    stats = {
        "correct_count": len(correct_conf),
        "incorrect_count": len(incorrect_conf),
        "correct_mean": float(np.mean(correct_conf)) if len(correct_conf) > 0 else 0.0,
        "correct_std": float(np.std(correct_conf)) if len(correct_conf) > 0 else 0.0,
        "incorrect_mean": float(np.mean(incorrect_conf)) if len(incorrect_conf) > 0 else 0.0,
        "incorrect_std": float(np.std(incorrect_conf)) if len(incorrect_conf) > 0 else 0.0
    }
    
    logger.info(
        f"Correct conf: {stats['correct_mean']:.3f} | "
        f"Incorrect conf: {stats['incorrect_mean']:.3f}"
    )
    return stats
