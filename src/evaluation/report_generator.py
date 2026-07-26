from pathlib import Path
from typing import Dict, Any, List, Union
import json
import csv

from src.logger import get_logger

logger = get_logger("DocForge.EvalReportGen")

def generate_markdown_report(
    cls_metrics: Dict[str, Any],
    loc_metrics: Dict[str, Any],
    confidence_stats: Dict[str, Any],
    failure_stats: Dict[str, Any],
    robustness_results: Dict[str, List[float]],
    config_dict: Dict[str, Any],
    save_path: Union[str, Path]
) -> None:
    """Generate a comprehensive human-readable Markdown evaluation report.

    Args:
        cls_metrics: Classification metrics dictionary.
        loc_metrics: Localization metrics dictionary.
        confidence_stats: Calibration and confidence statistics.
        failure_stats: Categorized failure profile statistics.
        robustness_results: Accuracy lists per perturbation level.
        config_dict: Experiment baseline parameters.
        save_path: Destination filepath of the report (.md).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract details
    tn = cls_metrics.get("confusion_matrix", {}).get("tn", 0)
    fp = cls_metrics.get("confusion_matrix", {}).get("fp", 0)
    fn = cls_metrics.get("confusion_matrix", {}).get("fn", 0)
    tp = cls_metrics.get("confusion_matrix", {}).get("tp", 0)
    
    total_samples = tn + fp + fn + tp
    
    md_content = f"""# DocForge - Evaluation & Benchmarking Report

This document reports the performance, localization quality, explanation outputs, calibration accuracy, and robustness characteristics of the fine-tuned Qwen2-VL model on the DocTamper dataset.

---

## 1. Model & Experiment Configurations

- **Base VLM Model:** `{config_dict.get("base_model_name", "Qwen/Qwen2-VL-2B-Instruct")}`
- **LoRA Rank (r):** `{config_dict.get("lora_rank", 8)}`
- **LoRA Alpha:** `{config_dict.get("lora_alpha", 16)}`
- **Evaluation Subset size:** `{total_samples} samples`
- **Device type:** `{config_dict.get("device", "cpu")}`
- **Autocast Precision:** `{config_dict.get("precision", "fp32")}`

---

## 2. Classification Metrics

The model's classification performance on forgery detection (Authentic vs. Tampered) is summarized below:

| Metric | Score / Value |
| :--- | :--- |
| **Accuracy** | {cls_metrics.get("accuracy", 0.0):.5f} |
| **Precision** | {cls_metrics.get("precision", 0.0):.5f} |
| **Recall (Sensitivity)** | {cls_metrics.get("recall", 0.0):.5f} |
| **F1-Score** | {cls_metrics.get("f1_score", 0.0):.5f} |
| **Balanced Accuracy** | {cls_metrics.get("balanced_accuracy", 0.0):.5f} |
| **ROC-AUC** | {cls_metrics.get("roc_auc", 0.5):.5f} |

### Confusion Matrix counts
- **True Negatives (Authentic $\\rightarrow$ Authentic):** `{tn:,}`
- **False Positives (Authentic $\\rightarrow$ Tampered):** `{fp:,}`
- **False Negatives (Tampered $\\rightarrow$ Authentic):** `{fn:,}`
- **True Positives (Tampered $\\rightarrow$ Tampered):** `{tp:,}`

---

## 3. Localization Metrics

Bounding-box localization of tampered document regions is evaluated at progressive overlap thresholds:

| Metric | Score |
| :--- | :--- |
| **Average IoU** | {loc_metrics.get("average_iou", 0.0):.5f} |
| **mAP (mean Average Precision)** | {loc_metrics.get("mAP", 0.0):.5f} |

### Detail thresholds scores

| Threshold (IoU $\\ge \\tau$) | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **IoU $\\ge$ 0.3** | {loc_metrics.get("precision_at_iou", {}).get("iou_0.3", 0.0):.4f} | {loc_metrics.get("recall_at_iou", {}).get("iou_0.3", 0.0):.4f} | {loc_metrics.get("f1_at_iou", {}).get("iou_0.3", 0.0):.4f} |
| **IoU $\\ge$ 0.5** | {loc_metrics.get("precision_at_iou", {}).get("iou_0.5", 0.0):.4f} | {loc_metrics.get("recall_at_iou", {}).get("iou_0.5", 0.0):.4f} | {loc_metrics.get("f1_at_iou", {}).get("iou_0.5", 0.0):.4f} |
| **IoU $\\ge$ 0.7** | {loc_metrics.get("precision_at_iou", {}).get("iou_0.7", 0.0):.4f} | {loc_metrics.get("recall_at_iou", {}).get("iou_0.7", 0.0):.4f} | {loc_metrics.get("f1_at_iou", {}).get("iou_0.7", 0.0):.4f} |

---

## 4. Confidence Analysis & Calibration

- **Expected Calibration Error (ECE):** `{confidence_stats.get("ece", 0.0):.5f}`
- **Mean confidence on Correct predictions:** `{confidence_stats.get("correct_mean", 0.0):.4f} (std={confidence_stats.get("correct_std", 0.0):.4f})`
- **Mean confidence on Incorrect predictions:** `{confidence_stats.get("incorrect_mean", 0.0):.4f} (std={confidence_stats.get("incorrect_std", 0.0):.4f})`

---

## 5. Failure Case Profiling

A total of `{failure_stats.get("total_failures", 0)}` failure cases were detected and clustered:

| Error Category | Case Count | Description |
| :--- | :--- | :--- |
| **False Positives** | {failure_stats.get("false_positives", 0)} | Authentic documents classified as tampered |
| **False Negatives** | {failure_stats.get("false_negatives", 0)} | Forged documents classified as authentic |
| **Localization Failures** | {failure_stats.get("localization_failures", 0)} | Correctly classified but bounding box IoU < 0.5 |
| **Low Confidence (Correct)** | {failure_stats.get("low_confidence_correct", 0)} | Correct prediction but confidence < 0.65 |

### Failures by Forgery Types
"""
    for forgery_type, count in failure_stats.get("failures_by_forgery_type", {}).items():
        md_content += f"- **{forgery_type}:** `{count} error cases`\n"

    md_content += """
---

## 6. Robustness Under Document Degradation

Accuracy decays of the model under progressively harder perturbations:

| Perturbation | Severity 0 (Clean) | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    severities = robustness_results.get("severities", [0, 1, 2, 3, 4, 5])
    for key, values in robustness_results.items():
        if key == "severities":
            continue
        row = f"| **{key.capitalize()}** "
        for v in values:
            row += f"| {v:.4f} "
        row += "|\n"
        md_content += row

    md_content += """
---

## 7. Generated Visualizations

The following high-resolution charts have been exported to `outputs/plots/`:
- **Confusion Matrix Heatmap:** `confusion_matrix.png`
- **ROC Curves:** `roc_curve.png`
- **Precision-Recall Curves:** `precision_recall_curve.png`
- **Reliability Bin Diagram:** `calibration_reliability.png`
- **Robustness Accuracy Decay:** `robustness_curves.png`
"""
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    logger.info(f"Generated human-readable Markdown report at {save_path}")
