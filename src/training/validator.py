import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

from src.training.utils import prepare_batch
from src.logger import get_logger

logger = get_logger("DocForge.Validator")

def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    processor: Any,
    loss_registry: nn.Module,
    device: torch.device,
    precision_dtype: torch.dtype = torch.float32
) -> Dict[str, float]:
    """Execute evaluation metrics over the validation dataset split.

    Disables gradient computation to prevent memory growth and runs in autocast.

    Args:
        model: VLM model to validate.
        val_loader: Validation DataLoader.
        processor: Qwen2VLDataProcessor instance.
        loss_registry: Loss manager registry.
        device: CUDA/CPU/MPS device.
        precision_dtype: Computation precision dtype.

    Returns:
        Dict[str, float]: Average loss metrics mapping.
    """
    model.eval()
    
    total_loss = 0.0
    loss_components: Dict[str, float] = {}
    num_batches = len(val_loader)
    
    if num_batches == 0:
        logger.warning("Empty validation loader. Skipping validation step.")
        return {"loss": 0.0}

    logger.info(f"Running validation loop on {num_batches} batches...")
    
    # Auto-resolve autocast device string
    device_type = device.type
    if device_type not in ["cuda", "cpu"]:
        autocast_enabled = False
    else:
        autocast_enabled = (precision_dtype != torch.float32)

    with torch.no_grad():
        for batch in val_loader:
            # Process raw batch into model-ready tensors on target device
            batch = prepare_batch(batch, processor, device)
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]
            
            with torch.amp.autocast(device_type=device_type, dtype=precision_dtype, enabled=autocast_enabled):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw
                )
                loss, loss_dict = loss_registry(outputs, batch)
                
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0.0) + v
                
    # Average metrics
    avg_metrics = {"loss": total_loss / num_batches}
    for k, v in loss_components.items():
        avg_metrics[k] = v / num_batches
        
    return avg_metrics
