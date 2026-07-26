import torch
from torch.optim import AdamW
import torch.nn as nn
from typing import List, Dict, Any
from src.logger import get_logger

logger = get_logger("DocForge.Optimizer")

def get_optimizer(
    model: nn.Module,
    learning_rate: float = 0.0002,
    weight_decay: float = 0.01
) -> AdamW:
    """Instantiate the AdamW optimizer, passing ONLY parameters that require gradients.

    Args:
        model: PEFT/LoRA model.
        learning_rate: Base learning rate.
        weight_decay: L2 regularization coefficient.

    Returns:
        AdamW: Configured optimizer.

    Raises:
        ValueError: If no trainable parameters are found.
    """
    logger.info("Initializing optimizer...")
    
    # Filter parameters
    trainable_params: List[nn.Parameter] = []
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    logger.info(f"Trainable parameters count: {trainable_count:,}")
    logger.info(f"Frozen parameters count: {frozen_count:,}")

    if len(trainable_params) == 0:
        logger.error("No trainable parameters found in model! Ensure LoRA adapters are attached.")
        raise ValueError("Cannot optimize model: Trainable parameter count is 0.")

    # Configure weight decay
    # Group parameters into decay vs no-decay if needed (bias/Norm layers have no decay)
    decay_params = []
    nodecay_params = []
    
    for param in trainable_params:
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)
            
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]

    optimizer = AdamW(
        optim_groups,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    logger.info(
        f"AdamW optimizer configured. Base LR: {learning_rate}, "
        f"Weight Decay: {weight_decay}. Groups: {len(optim_groups)}."
    )
    return optimizer
