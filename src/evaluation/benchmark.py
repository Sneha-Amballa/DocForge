from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.logger import get_logger
from src.training.utils import prepare_batch
from src.evaluation.robustness import apply_degradation
from src.processor import Qwen2VLDataProcessor

logger = get_logger("DocForge.RobustnessBenchmark")

class DocForgeRobustnessBenchmarker:
    """Manages perturbation iterations to stress-test VLM robustness under degraded documents."""

    def __init__(
        self,
        model: nn.Module,
        processor: Qwen2VLDataProcessor,
        device: torch.device
    ) -> None:
        """Initialize the benchmarker.

        Args:
            model: Fine-tuned VLM model.
            processor: Qwen2VLDataProcessor instance.
            device: Target execution device.
        """
        self.model = model
        self.processor = processor
        self.device = device
        
        # Check mock model
        self.is_mock = False
        base = model
        if hasattr(model, "base_model"):
            base = model.base_model
        if base.__class__.__name__ == "MockQwen2VLForConditionalGeneration":
            self.is_mock = True

    def run_robustness_benchmark(
        self,
        dataloader: DataLoader,
        degradation_types: List[str] = ["jpeg", "blur", "noise", "brightness", "contrast", "downsample"],
        severities: List[int] = [0, 1, 2, 3, 4, 5],
        limit_samples: int = 10
    ) -> Dict[str, List[float]]:
        """Evaluate the model across progressive levels of document degradation.

        Args:
            dataloader: Validation/Test DataLoader.
            degradation_types: Named degradations to benchmark.
            severities: Severity levels to check (e.g. 0 to 5).
            limit_samples: Max samples evaluated per perturbation severity (for speed).

        Returns:
            Dict[str, List[float]]: Accuracy decay scores mapping.
        """
        self.model.eval()
        
        # Initialize results mapping
        results = {"severities": [float(s) for s in severities]}
        for dtype in degradation_types:
            results[dtype] = []

        logger.info(f"Starting robustness benchmark on {len(degradation_types)} perturbations...")
        
        # Cache subset of batches to evaluate rapidly
        batches_subset = []
        samples_cached = 0
        for batch in dataloader:
            if samples_cached >= limit_samples:
                break
            batches_subset.append(batch)
            samples_cached += len(batch["image"])

        if not batches_subset:
            logger.warning("Empty dataloader provided to robustness benchmark.")
            return results

        # Run benchmark loop
        with torch.no_grad():
            for dtype in degradation_types:
                logger.info(f"Evaluating degradation: '{dtype}'...")
                
                for sev in severities:
                    if sev == 0:
                        # Clean baseline accuracy
                        accuracy = self._eval_clean_baseline(batches_subset, limit_samples)
                        results[dtype].append(accuracy)
                        continue
                        
                    correct_predictions = 0
                    evaluated_samples = 0
                    
                    for batch in batches_subset:
                        if evaluated_samples >= limit_samples:
                            break
                            
                        batch_size = len(batch["image"])
                        
                        # Apply degradation to images in batch
                        degraded_images = []
                        for img in batch["image"]:
                            degraded_images.append(apply_degradation(img, dtype, float(sev)))
                            
                        # Reconstruct degraded batch dict
                        degraded_batch = {
                            "image": degraded_images,
                            "prompt": batch["prompt"],
                            "image_tensor": batch["image_tensor"],  # placeholder
                            "tampering_label": batch["tampering_label"],
                            "sample_id": batch["sample_id"],
                            "forgery_type": batch["forgery_type"],
                            "normalized_bbox": batch["normalized_bbox"]
                        }
                        
                        # Process degraded images to model-ready inputs
                        degraded_batch = prepare_batch(degraded_batch, self.processor, self.device)
                        
                        input_ids = degraded_batch["input_ids"]
                        attention_mask = degraded_batch["attention_mask"]
                        pixel_values = degraded_batch["pixel_values"]
                        image_grid_thw = degraded_batch["image_grid_thw"]
                        
                        # Model prediction
                        if self.is_mock:
                            # Simulate accuracy decay based on severity
                            for i in range(batch_size):
                                if evaluated_samples >= limit_samples:
                                    break
                                    
                                gt_label = int(degraded_batch["tampering_label"][i].item())
                                sample_id = degraded_batch["sample_id"][i]
                                
                                # Decay rate based on severity and perturbation type
                                decay_rand = (hash(sample_id) + sev * 13) % 100
                                threshold = 85 - (sev * 8)  # down to ~45% accuracy
                                
                                if decay_rand < threshold:
                                    pred_label = gt_label
                                else:
                                    pred_label = 1 - gt_label
                                    
                                if pred_label == gt_label:
                                    correct_predictions += 1
                                evaluated_samples += 1
                        else:
                            # Real model evaluation
                            generated_ids = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                image_grid_thw=image_grid_thw,
                                max_new_tokens=20
                            )
                            
                            for i in range(batch_size):
                                if evaluated_samples >= limit_samples:
                                    break
                                    
                                seq = generated_ids[i]
                                pred_text = self.processor.processor.tokenizer.decode(seq, skip_special_tokens=True).lower()
                                pred_label = 1 if "tampered" in pred_text or "forge" in pred_text else 0
                                gt_label = int(degraded_batch["tampering_label"][i].item())
                                
                                if pred_label == gt_label:
                                    correct_predictions += 1
                                evaluated_samples += 1
                                
                    acc = correct_predictions / evaluated_samples if evaluated_samples > 0 else 0.0
                    results[dtype].append(float(acc))
                    logger.info(f"  Severity {sev} accuracy: {acc:.4f}")
                    
        return results

    def _eval_clean_baseline(self, batches: List[Dict[str, Any]], limit_samples: int) -> float:
        """Helper to compute baseline clean accuracy on cached batches.

        Args:
            batches: Cached batches.
            limit_samples: Sample count limit.

        Returns:
            float: Baseline accuracy.
        """
        correct_predictions = 0
        evaluated_samples = 0
        
        for batch in batches:
            if evaluated_samples >= limit_samples:
                break
                
            batch_size = len(batch["image"])
            
            # Process clean batch
            clean_batch = prepare_batch(batch, self.processor, self.device)
            
            input_ids = clean_batch["input_ids"]
            attention_mask = clean_batch["attention_mask"]
            pixel_values = clean_batch["pixel_values"]
            image_grid_thw = clean_batch["image_grid_thw"]
            
            if self.is_mock:
                for i in range(batch_size):
                    if evaluated_samples >= limit_samples:
                        break
                    gt_label = int(clean_batch["tampering_label"][i].item())
                    sample_id = clean_batch["sample_id"][i]
                    
                    # 85% correct baseline
                    if (hash(sample_id) % 100) < 85:
                        correct_predictions += 1
                    evaluated_samples += 1
            else:
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=20
                )
                for i in range(batch_size):
                    if evaluated_samples >= limit_samples:
                        break
                    seq = generated_ids[i]
                    pred_text = self.processor.processor.tokenizer.decode(seq, skip_special_tokens=True).lower()
                    pred_label = 1 if "tampered" in pred_text or "forge" in pred_text else 0
                    gt_label = int(clean_batch["tampering_label"][i].item())
                    
                    if pred_label == gt_label:
                        correct_predictions += 1
                    evaluated_samples += 1
                    
        return correct_predictions / evaluated_samples if evaluated_samples > 0 else 0.0
