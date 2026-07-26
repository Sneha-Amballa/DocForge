import time
from typing import Dict, Any, List, Union
from PIL import Image
import torch
import torch.nn as nn

from src.logger import get_logger
from src.processor import Qwen2VLDataProcessor
from src.training.utils import prepare_batch
from src.deployment.inference.postprocess import postprocess_forgery_response

logger = get_logger("DocForge.DeployPredictor")

class DocForgePredictor:
    """Production predictor wrapper executing real-time document validation checks."""

    def __init__(
        self,
        model: nn.Module,
        processor: Qwen2VLDataProcessor,
        device: torch.device
    ) -> None:
        """Initialize the predictor.

        Args:
            model: Cached model instance.
            processor: AutoProcessor wrapper instance.
            device: Execution device.
        """
        self.model = model
        self.processor = processor
        self.device = device
        
        # Detect mock model
        self.is_mock = False
        base = model
        if hasattr(model, "base_model"):
            base = model.base_model
        if base.__class__.__name__ == "MockQwen2VLForConditionalGeneration":
            from src.config import DatasetConfig
            cfg = DatasetConfig()
            if not cfg.dry_run:
                raise RuntimeError("Critical: Cannot run deployment with mock model when dry_run is set to False.")
            self.is_mock = True
            logger.info("Predictor running in mock verification mode.")

    def predict_image(
        self,
        image: Image.Image,
        prompt: str = "Assess if this document contains any tampered or forged areas."
    ) -> Dict[str, Any]:
        """Perform document forgery detection on a single uploaded image.

        Args:
            image: PIL Image object.
            prompt: Text prompt instruction for the model.

        Returns:
            Dict[str, Any]: Formatted json output containing prediction metrics.
        """
        start_time = time.perf_counter()
        
        img_width, img_height = image.size
        
        # Handle predictions
        if self.is_mock:
            # Deterministic mock prediction based on prompt or image size
            # If prompt mentions authentic, simulate authentic. Else tampered.
            prompt_lower = prompt.lower()
            tampered = True
            if "authentic" in prompt_lower or "clean" in prompt_lower:
                tampered = False
                
            # Simulate latency 150ms to 300ms
            time.sleep(0.2)
            
            if tampered:
                # Mock coords: [ymin, xmin, ymax, xmax] -> [284, 112, 326, 240] (normalized)
                ymin, xmin, ymax, xmax = 284, 112, 326, 240
                
                # Scale coordinates
                xmin_abs = int(round(xmin * img_width / 1000.0))
                ymin_abs = int(round(ymin * img_height / 1000.0))
                xmax_abs = int(round(xmax * img_width / 1000.0))
                ymax_abs = int(round(ymax * img_height / 1000.0))
                
                bounding_boxes = [{
                    "x": xmin_abs,
                    "y": ymin_abs,
                    "width": xmax_abs - xmin_abs,
                    "height": ymax_abs - ymin_abs
                }]
                
                forgery_type = "Text Replacement"
                explanation = f"Detected digital editing/modification in text box {bounding_boxes}."
                confidence = 0.97
            else:
                bounding_boxes = []
                forgery_type = "Authentic"
                explanation = "The document layout and text lines are consistent. No forgery detected."
                confidence = 0.99
                
            processing_time_ms = int((time.perf_counter() - start_time) * 1000.0)
            
            return {
                "tampered": tampered,
                "prediction": "Tampered" if tampered else "Authentic",
                "confidence": confidence,
                "forgery_type": forgery_type,
                "bounding_boxes": bounding_boxes,
                "explanation": explanation,
                "reasoning": explanation,
                "processing_time_ms": processing_time_ms,
                "latency_ms": processing_time_ms
            }

        # Real model inference pipeline
        try:
            # Create a mock batch structure to pass through prepare_batch
            # Because prepare_batch scales and stacks inputs correctly
            from src.config import DatasetConfig
            cfg = DatasetConfig()
            
            # Simple preprocess resize
            preprocessed_img = image.resize(cfg.image_size, Image.Resampling.BILINEAR)
            import numpy as np
            import torch
            img_array = np.array(preprocessed_img)
            # PIL image is (H, W, C), convert to (C, H, W)
            img_tensor = torch.tensor(img_array.transpose((2, 0, 1)), dtype=torch.float32) / 255.0
            
            # Build mock batch
            batch = {
                "image": [image],
                "prompt": [prompt],
                "image_tensor": img_tensor.unsqueeze(0),
                "tampering_label": torch.tensor([0]),
                "sample_id": ["upload_sample"],
                "forgery_type": ["Unknown"],
                "normalized_bbox": [[]]
            }
            
            # Prepare batch tensors
            batch = prepare_batch(batch, self.processor, self.device)
            
            # Generate sequence with logits scores
            import numpy as np
            generation_outputs = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                max_new_tokens=100,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            # Extract generated tokens and scores
            generated_ids = generation_outputs.sequences
            seq = generated_ids[0]
            
            prompt_len = batch["input_ids"].shape[1]
            pred_ids = seq[prompt_len:]
            pred_text = self.processor.processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
            
            # Parse responses
            tampered, forgery_type, bounding_boxes, explanation = postprocess_forgery_response(
                generated_text=pred_text,
                img_width=img_width,
                img_height=img_height
            )
            
            # Compute confidence from logits softmax
            scores = generation_outputs.scores  # Tuple of shape (batch_size, vocab_size) at each gen step
            probs_list = []
            if scores is not None and len(scores) > 0:
                for step_idx in range(min(len(scores), len(pred_ids))):
                    step_logits = scores[step_idx]  # shape (batch_size, vocab_size)
                    step_probs = torch.softmax(step_logits, dim=-1)
                    
                    token_id = pred_ids[step_idx].item()
                    if token_id < step_probs.shape[1]:
                        token_prob = step_probs[0, token_id].item()
                        probs_list.append(token_prob)
                        
            if probs_list:
                confidence = float(np.mean(probs_list)) * 100.0  # Normalize to 0-100%
            else:
                confidence = 95.0
            
            processing_time_ms = int((time.perf_counter() - start_time) * 1000.0)
            
            return {
                "tampered": tampered,
                "prediction": "Tampered" if tampered else "Authentic",
                "confidence": confidence,
                "forgery_type": forgery_type,
                "bounding_boxes": bounding_boxes,
                "explanation": explanation,
                "reasoning": explanation,
                "processing_time_ms": processing_time_ms,
                "latency_ms": processing_time_ms
            }
        except Exception as e:
            logger.error(f"Inference pipeline execution crashed: {e}")
            raise RuntimeError(f"Model prediction failed: {e}")
