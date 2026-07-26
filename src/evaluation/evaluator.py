import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from src.logger import get_logger
from src.training.utils import prepare_batch
from src.processor import Qwen2VLDataProcessor

logger = get_logger("DocForge.Evaluator")

def parse_boxes_from_text(text: str) -> List[List[int]]:
    """Parse bounding boxes from Qwen2-VL format '[ymin, xmin, ymax, xmax]' text.

    Converts coordinates from [ymin, xmin, ymax, xmax] (scaled to 1000)
    to standard [xmin, ymin, xmax, ymax] scale.

    Args:
        text: Generated VLM output string.

    Returns:
        List[List[int]]: Standardized bounding box coordinates.
    """
    boxes = []
    # Match pattern like [ymin, xmin, ymax, xmax]
    matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    for m in matches:
        try:
            ymin, xmin, ymax, xmax = map(int, m)
            # Reorder coordinates to [xmin, ymin, xmax, ymax]
            boxes.append([xmin, ymin, xmax, ymax])
        except ValueError:
            pass
    return boxes

class DocForgeEvaluator:
    """Orchestrates predictions, coordinates parsing, and metrics collection over test sets."""

    def __init__(
        self,
        model: nn.Module,
        processor: Qwen2VLDataProcessor,
        device: torch.device
    ) -> None:
        """Initialize the evaluator.

        Args:
            model: Fine-tuned PEFT/LoRA model.
            processor: Qwen2VLDataProcessor instance.
            device: Target execution device.
        """
        self.model = model
        self.processor = processor
        self.device = device
        
        # Check if the model is a mock model
        self.is_mock = False
        base = model
        if hasattr(model, "base_model"):
            base = model.base_model
        if base.__class__.__name__ == "MockQwen2VLForConditionalGeneration":
            self.is_mock = True
            logger.info("Mock VLM detected. Evaluator will run in simulation verification mode.")

    def run_inference(
        self,
        dataloader: DataLoader,
        limit_samples: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[int], List[int], List[float], List[List[List[int]]], List[List[List[int]]]]:
        """Run batch inference over dataset split and extract metrics properties.

        Args:
            dataloader: Test dataset DataLoader.
            limit_samples: Max number of samples to evaluate (optional).

        Returns:
            Tuple containing:
                - samples_metadata: List of metadata dictionaries.
                - y_true: List of ground-truth labels.
                - y_pred: List of predicted labels.
                - y_prob: List of predicted confidence scores.
                - gt_boxes_list: List of ground-truth box lists.
                - pred_boxes_list: List of predicted box lists.
        """
        self.model.eval()
        
        samples_metadata = []
        y_true = []
        y_pred = []
        y_prob = []
        gt_boxes_list = []
        pred_boxes_list = []
        
        samples_count = 0
        
        logger.info(f"Starting evaluation run (Mock simulation mode: {self.is_mock})...")
        
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                batch_size = len(batch["image"])
                
                # Check sample count limit
                if limit_samples is not None and samples_count >= limit_samples:
                    break

                # Prepare VLM tensors
                batch = prepare_batch(batch, self.processor, self.device)
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]
                
                # Model forward/inference
                if self.is_mock:
                    # In mock mode, simulate realistic prediction errors to verify failure clusters
                    # Accuracy target: ~80%
                    for i in range(batch_size):
                        if limit_samples is not None and samples_count >= limit_samples:
                            break
                            
                        gt_label = int(batch["tampering_label"][i].item())
                        sample_id = batch["sample_id"][i]
                        forgery_type = batch["forgery_type"][i]
                        
                        # Apply deterministic noise based on sample_id hash
                        seed_val = hash(sample_id) % 100
                        
                        # 80% correct labels
                        if seed_val < 80:
                            pred_label = gt_label
                        else:
                            pred_label = 1 - gt_label
                            
                        # Correct confidences range [0.70, 0.99], incorrect [0.52, 0.72]
                        if pred_label == gt_label:
                            conf = 0.70 + (seed_val % 30) / 100.0
                        else:
                            conf = 0.51 + (seed_val % 20) / 100.0
                            
                        # Map probability value corresponding to positive class
                        prob = conf if pred_label == 1 else 1.0 - conf
                        
                        gt_boxes = batch["normalized_bbox"][i]
                        pred_boxes = []
                        
                        if pred_label == 1:
                            if gt_label == 1 and len(gt_boxes) > 0:
                                # For true positives, simulate box matches
                                # 75% overlap match (TP), 25% box misses (Localization Failure)
                                if seed_val % 4 != 0:
                                    # Create overlapping box
                                    for g_box in gt_boxes:
                                        p_box = [
                                            max(0, g_box[0] - (seed_val % 10)),
                                            max(0, g_box[1] - (seed_val % 10)),
                                            min(1000, g_box[2] + (seed_val % 10)),
                                            min(1000, g_box[3] + (seed_val % 10))
                                        ]
                                        pred_boxes.append(p_box)
                                else:
                                    # Random box mismatch (Loc Fail)
                                    pred_boxes.append([10, 10, 100, 100])
                            else:
                                # False alarm box prediction
                                pred_boxes.append([200, 200, 400, 400])
                                
                        explanation = f"Generated explanation for sample {sample_id}: "
                        if pred_label == 1:
                            explanation += f"Tampering detected in document region {pred_boxes}."
                        else:
                            explanation += "No document forgery detected."

                        # Record outputs
                        y_true.append(gt_label)
                        y_pred.append(pred_label)
                        y_prob.append(prob)
                        gt_boxes_list.append(gt_boxes)
                        pred_boxes_list.append(pred_boxes)
                        
                        samples_metadata.append({
                            "sample_id": sample_id,
                            "forgery_type": forgery_type,
                            "explanation": explanation
                        })
                        
                        samples_count += 1
                else:
                    # REAL Model inference
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw
                    )
                    
                    # Generate output sequences
                    # Using greedy decoding
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        max_new_tokens=100
                    )
                    
                    # Logits mapping for confidence
                    logits = outputs.logits
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    
                    for i in range(batch_size):
                        if limit_samples is not None and samples_count >= limit_samples:
                            break
                            
                        # Retrieve text
                        seq = generated_ids[i]
                        pred_text = self.processor.processor.tokenizer.decode(seq, skip_special_tokens=True).lower()
                        
                        # Parse predicted label
                        pred_label = 1 if "tampered" in pred_text or "forge" in pred_text else 0
                        
                        # Retrieve confidence from logits
                        # Softmax score of label token or mean prob
                        prob = float(probs[i].max().cpu().item())
                        
                        # Parse predicted boxes
                        pred_boxes = parse_boxes_from_text(pred_text)
                        
                        # Ground truth
                        gt_label = int(batch["tampering_label"][i].item())
                        gt_boxes = batch["normalized_bbox"][i]
                        
                        y_true.append(gt_label)
                        y_pred.append(pred_label)
                        y_prob.append(prob)
                        gt_boxes_list.append(gt_boxes)
                        pred_boxes_list.append(pred_boxes)
                        
                        samples_metadata.append({
                            "sample_id": batch["sample_id"][i],
                            "forgery_type": batch["forgery_type"][i],
                            "explanation": pred_text
                        })
                        
                        samples_count += 1
                        
            logger.info(f"Completed batch inferences on {samples_count} samples.")
            
        return samples_metadata, y_true, y_pred, y_prob, gt_boxes_list, pred_boxes_list
