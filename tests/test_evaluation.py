import unittest
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is in system path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.localization import compute_bbox_iou, evaluate_localization_set
from src.evaluation.confidence import compute_calibration_error, analyze_confidence_distributions
from src.evaluation.robustness import apply_degradation

class TestEvaluationPipeline(unittest.TestCase):
    """Unit tests verifying Phase 4 metrics calculations and image degradations."""

    def test_classification_metrics(self) -> None:
        """Test accuracy, precision, recall, and f1 score math."""
        # Simple balanced labels
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        y_prob = [0.1, 0.9, 0.8, 0.2]
        
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("roc_auc", metrics)
        
        # accuracy = 2 / 4 = 0.5
        self.assertEqual(metrics["accuracy"], 0.5)
        # confusion matrix counts
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["tn"], 1)
        self.assertEqual(cm["fp"], 1)
        self.assertEqual(cm["fn"], 1)
        self.assertEqual(cm["tp"], 1)

    def test_bbox_iou(self) -> None:
        """Test Intersection over Union bounding box overlaps."""
        # Identical boxes
        boxA = [10, 10, 100, 100]
        boxB = [10, 10, 100, 100]
        self.assertAlmostEqual(compute_bbox_iou(boxA, boxB), 1.0)
        
        # Mismatch
        boxC = [200, 200, 300, 300]
        self.assertAlmostEqual(compute_bbox_iou(boxA, boxC), 0.0)
        
        # Partial overlap
        boxD = [50, 50, 150, 150]
        # Box A area = 90 * 90 = 8100
        # Box D area = 100 * 100 = 10000
        # Intersection = [50, 50, 100, 100] -> area = 50 * 50 = 2500
        # Union = 8100 + 10000 - 2500 = 15600
        # IoU = 2500 / 15600 = 0.16025
        self.assertAlmostEqual(compute_bbox_iou(boxA, boxD), 2500 / 15600, places=4)

    def test_ece_calculation(self) -> None:
        """Test ECE and calibration bin math."""
        y_true = [0, 1, 0, 1]
        y_prob = [0.1, 0.9, 0.8, 0.2]
        
        ece_data = compute_calibration_error(y_true, y_prob, num_bins=5)
        self.assertIn("ece", ece_data)
        self.assertIn("bins", ece_data)
        self.assertLessEqual(ece_data["ece"], 1.0)
        self.assertGreaterEqual(ece_data["ece"], 0.0)
        
        stats = analyze_confidence_distributions(y_true, y_prob)
        self.assertIn("correct_mean", stats)
        self.assertIn("incorrect_mean", stats)

    def test_degradations(self) -> None:
        """Test PIL image degradation filters."""
        # Create solid mock image
        img = Image.new("RGB", (100, 100), color="white")
        
        # Apply JPEG degradation
        jpeg_img = apply_degradation(img, "jpeg", severity=3)
        self.assertEqual(jpeg_img.size, (100, 100))
        
        # Apply Gaussian noise
        noise_img = apply_degradation(img, "noise", severity=2)
        self.assertEqual(noise_img.size, (100, 100))
        
        # Apply Gaussian blur
        blur_img = apply_degradation(img, "blur", severity=4)
        self.assertEqual(blur_img.size, (100, 100))

if __name__ == "__main__":
    unittest.main()
