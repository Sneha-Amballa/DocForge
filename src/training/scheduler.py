import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from src.logger import get_logger

logger = get_logger("DocForge.Scheduler")

def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_warmup_steps: int = 100,
    num_training_steps: int = 1000
) -> LambdaLR:
    """Construct a configurable learning rate scheduler with optional warmup.

    Supports 'cosine', 'linear', and 'constant' schedules.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_type: String identifier ('cosine', 'linear', 'constant').
        num_warmup_steps: Number of initial warmup steps.
        num_training_steps: Total steps in training.

    Returns:
        LambdaLR: Configured scheduler.
    """
    logger.info(
        f"Creating scheduler: Type={scheduler_type}, Warmup={num_warmup_steps} steps, "
        f"Total={num_training_steps} steps."
    )
    
    stype = scheduler_type.lower().strip()

    if stype == "cosine":
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            # Clip progress to [0.0, 1.0]
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
    elif stype == "linear":
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            # Clip progress to [0.0, 1.0]
            progress = min(1.0, max(0.0, progress))
            return max(0.0, 1.0 - progress)
            
    else:  # 'constant'
        if stype != "constant":
            logger.warning(f"Unknown scheduler type '{scheduler_type}'. Defaulting to constant scheduler.")
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler
