import torch
import torch.nn as nn
from typing import Dict, Any, List
from src.model.utils import set_seed as base_set_seed
from src.processor import Qwen2VLDataProcessor

def set_seed(seed: int = 42) -> None:
    """Set global seeds for reproducibility."""
    base_set_seed(seed)

def get_gpu_memory_usage() -> str:
    """Get a formatted string of the current CUDA GPU memory allocation.

    Returns:
        str: Memory usage string.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        return f"{allocated:.2f}GB / {reserved:.2f}GB (Alloc/Res)"
    return "N/A (CPU/MPS)"

def prepare_batch(
    batch: Dict[str, Any],
    processor: Qwen2VLDataProcessor,
    device: torch.device
) -> Dict[str, Any]:
    """Process a batch of dataset samples through the processor, stacking and padding tensors.

    This ensures that variable length prompts/tokenizer token counts are correctly
    padded, and all tensors are mapped to the target execution device.

    Args:
        batch: Batch dict from the DataLoader.
        processor: Qwen2VLDataProcessor instance.
        device: Target execution device.

    Returns:
        Dict[str, Any]: Updated batch dict with model-ready tensor inputs:
                        'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'.
    """
    batch_size = len(batch["image"])
    
    # Process each sample individually
    samples_inputs = []
    for idx in range(batch_size):
        sample = {
            "image": batch["image"][idx],
            "prompt": batch["prompt"][idx],
            "image_tensor": batch["image_tensor"][idx],
        }
        processed = processor.process_sample(sample)
        samples_inputs.append(processed)
        
    # Stack / Pad input_ids and attention_mask
    input_ids_list = [s["input_ids"][0] for s in samples_inputs]
    attention_mask_list = [s["attention_mask"][0] for s in samples_inputs]
    
    max_len = max(len(ids) for ids in input_ids_list)
    
    # Resolve pad token id
    pad_token_id = 0
    if not processor.is_fallback and processor.processor is not None:
        if hasattr(processor.processor, "tokenizer") and hasattr(processor.processor.tokenizer, "pad_token_id"):
            if processor.processor.tokenizer.pad_token_id is not None:
                pad_token_id = processor.processor.tokenizer.pad_token_id
                
    padded_input_ids = torch.stack([
        torch.cat([ids, torch.full((max_len - len(ids),), pad_token_id, dtype=torch.long)])
        for ids in input_ids_list
    ]).to(device)
    
    padded_attention_mask = torch.stack([
        torch.cat([m, torch.zeros(max_len - len(m), dtype=torch.long)])
        for m in attention_mask_list
    ]).to(device)
    
    # Concatenate pixel_values
    pixel_values_list = [s["pixel_values"] for s in samples_inputs]
    # For Qwen2-VL HF Processor, pixel_values is 2D (num_patches, dim) and must be concatenated.
    # For fallback, it is (1, 3, H, W) and also can be concatenated.
    if len(pixel_values_list) > 0:
        pixel_values = torch.cat(pixel_values_list, dim=0).to(device)
    else:
        pixel_values = None
        
    image_grid_thw = torch.cat([s["image_grid_thw"] for s in samples_inputs], dim=0).to(device)
    
    # Add model-ready tensors to batch dictionary
    batch["input_ids"] = padded_input_ids
    batch["attention_mask"] = padded_attention_mask
    batch["pixel_values"] = pixel_values
    batch["image_grid_thw"] = image_grid_thw
    
    return batch
