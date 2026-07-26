import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np

from src.logger import get_logger

logger = get_logger("DocForge.EvalPlots")

def plot_confusion_matrix(
    tn: int, fp: int, fn: int, tp: int,
    save_path: Union[str, Path]
) -> None:
    """Plot and save a neat 2x2 confusion matrix heatmap.

    Args:
        tn: True negatives.
        fp: False positives.
        fn: False negatives.
        tp: True positives.
        save_path: Output PNG filepath.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    matrix = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    
    plt.colorbar(im)
    
    # Class labels
    classes = ["Authentic", "Tampered"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    
    # Label cell counts
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            # Use white color if cell is dark
            color = "white" if val > (matrix.max() / 2.0) else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")
            
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix chart to {save_path}")

def plot_roc_pr_curves(
    y_true: List[int],
    y_prob: List[float],
    save_dir: Union[str, Path]
) -> None:
    """Plot and save ROC and Precision-Recall curves.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted confidence probabilities.
        save_dir: Folder path where plots are written.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    
    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    except Exception:
        # Simple fallback baseline
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random (AUC = 0.500)")
        
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    fig.savefig(save_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color="green", lw=2, label=f"PR Curve (AP = {ap:.3f})")
    except Exception:
        # baseline
        ax.plot([0, 1], [0.5, 0.5], color="navy", lw=2, linestyle="--", label="Baseline")
        
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    fig.savefig(save_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ROC and PR curves in {save_dir}")

def plot_calibration_diagram(
    ece_data: Dict[str, Any],
    save_path: Union[str, Path]
) -> None:
    """Plot and save a Reliability Diagram showing model calibration bins.

    Args:
        ece_data: Dictionary returned by compute_calibration_error.
        save_path: Output PNG filepath.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    ece = ece_data.get("ece", 0.0)
    bins = ece_data.get("bins", [])
    
    if not bins:
        return
        
    # Extract data
    bin_accs = [b["accuracy"] for b in bins]
    bin_confs = [b["confidence"] for b in bins]
    bin_labels = [f"{b['range'][0]:.1f}-{b['range'][1]:.1f}" for b in bins]
    x = np.arange(len(bins))
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Draw bars representing accuracy
    ax.bar(x - 0.2, bin_accs, width=0.4, label="Accuracy", color="salmon", align="center")
    # Draw bars representing confidence
    ax.bar(x + 0.2, bin_confs, width=0.4, label="Confidence", color="skyblue", align="center")
    
    # Perfect calibration line
    ax.plot([-0.5, len(bins) - 0.5], [0.05, 0.95], color="gray", linestyle="--", label="Perfect Calibration")
    
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_xlabel("Confidence Interval", fontsize=11)
    ax.set_ylabel("Ratio", fontsize=11)
    ax.set_title(f"Reliability Diagram (ECE = {ece:.5f})", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Reliability Diagram to {save_path}")

def plot_robustness_curves(
    robustness_results: Dict[str, List[float]],
    save_path: Union[str, Path]
) -> None:
    """Plot accuracy decay curves across different document perturbations.

    Args:
        robustness_results: Dictionary containing 'severities' and performance lists.
            Example: {'severities': [0, 1, 2, 3], 'blur': [0.95, 0.90, 0.82, 0.70]}
        save_path: Output PNG filepath.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    severities = robustness_results.get("severities", [])
    if not severities:
        return
        
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ["red", "blue", "green", "orange", "purple", "brown", "magenta"]
    color_idx = 0
    
    for key, values in robustness_results.items():
        if key == "severities":
            continue
            
        ax.plot(
            severities,
            values,
            marker="o",
            linewidth=2,
            label=key.capitalize(),
            color=colors[color_idx % len(colors)]
        )
        color_idx += 1
        
    ax.set_xlabel("Perturbation Severity Level", fontsize=11, fontweight="bold")
    ax.set_ylabel("Model Accuracy Score", fontsize=11, fontweight="bold")
    ax.set_title("Robustness Performance Decay", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(severities)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved robustness curves to {save_path}")
