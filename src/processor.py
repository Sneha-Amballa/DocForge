from typing import Dict, Any, List, Optional, Union
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import torch

from src.logger import get_logger

logger = get_logger("DocForge.Processor")

class Qwen2VLDataProcessor:
    """Wrapper class for Hugging Face AutoProcessor for Qwen2-VL models.

    Prepares unified dataset samples into tensors compatible with Qwen2-VL models.
    Supports offline/fallback processing modes if the Hugging Face weights or packages cannot be loaded.
    """

    def __init__(self, processor_name: str = "Qwen/Qwen2-VL-2B-Instruct") -> None:
        """Initialize the processor.

        Args:
            processor_name: Hugging Face repository or path of the processor model.
        """
        self.processor_name = processor_name
        self.processor: Optional[Any] = None
        self.is_fallback = False

        self._load_processor()

    def _load_processor(self) -> None:
        """Attempt to load AutoProcessor from pretrained model path."""
        try:
            # pyrefly: ignore [missing-import]
            from transformers import AutoProcessor
            logger.info(f"Loading Qwen2-VL AutoProcessor from '{self.processor_name}'...")
            # Set trust_remote_code=True for Qwen models
            self.processor = AutoProcessor.from_pretrained(
                self.processor_name,
                trust_remote_code=True
            )
            logger.info("AutoProcessor loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load Hugging Face AutoProcessor for '{self.processor_name}': {e}. "
                "Switching to offline/fallback processor mode."
            )
            self.is_fallback = True
            self.processor = None

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single unified sample representation dict for Qwen2-VL.

        Formats image inputs and VLM prompt templates.

        Args:
            sample: Standardized sample dictionary from DocTamperTorchDataset.

        Returns:
            Dict[str, Any]: Dictionary containing processor-ready tensors:
                - If HF processor works: Hugging Face BatchEncoding tensors
                  ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', etc.)
                - If fallback: mock text tensors and preprocessed image tensors.
        """
        prompt_text = sample["prompt"]
        image = sample["image"]  # Preprocessed PIL image

        # Construct conversational message structure for Qwen2-VL
        # Each message contains text and image placeholders
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        if not self.is_fallback and self.processor is not None:
            try:
                # Apply VLM chat template to get instruction text
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Pass to processor to generate batch input tensors
                inputs = self.processor(
                    text=[text], 
                    images=[image], 
                    padding=True, 
                    return_tensors="pt"
                )
                
                # Convert BatchEncoding to normal dict
                inputs_dict = {k: v for k, v in inputs.items()}
                logger.debug("Successfully processed sample using Hugging Face AutoProcessor.")
                return inputs_dict
                
            except Exception as e:
                logger.error(f"Error during Hugging Face processor execution: {e}. Falling back.")
                # Continue to fallback implementation

        # Fallback implementation
        # Mock token indices: split prompt and assign dummy index per word
        words = prompt_text.split()
        mock_input_ids = torch.tensor([[100] + [hash(w) % 10000 for w in words] + [101]], dtype=torch.long)
        mock_attention_mask = torch.ones_like(mock_input_ids, dtype=torch.long)
        
        # Use pre-computed image tensor (unsqueezed to add batch dimension)
        pixel_values = sample["image_tensor"].unsqueeze(0)  # Shape: (1, 3, H, W)
        
        # Qwen2-VL grid dimensions (H/28, W/28)
        h, w = image.size
        grid_h = h // 28
        grid_w = w // 28
        image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
        
        logger.debug("Successfully processed sample using fallback processor mode.")
        return {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "is_fallback": True
        }
