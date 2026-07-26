import torch
import torch.nn as nn
from typing import Dict, Any, List

from src.model.lora import get_trainable_parameters_summary
from src.logger import get_logger

logger = get_logger("DocForge.ModelSummary")

def print_model_parameter_summary(model: nn.Module) -> None:
    """Print a detailed summary table of the model's layers and parameters.

    Iterates over all named parameters, displaying layer shape, parameter counts,
    and whether it is trainable or frozen.

    Args:
        model: PyTorch model.
    """
    lines = []
    lines.append("+" + "-"*80 + "+")
    lines.append(f"| {'Layer Name':<45} | {'Shape':<15} | {'Params':<10} | {'Trainable':<5} |")
    lines.append("+" + "-"*80 + "+")
    
    # We print the first 25 parameters to avoid spamming the log for huge models,
    # but still show the structural layout.
    params_list = list(model.named_parameters())
    num_params = len(params_list)
    limit = 30
    
    for idx, (name, param) in enumerate(params_list[:limit]):
        shape_str = str(list(param.shape))
        if len(shape_str) > 15:
            shape_str = shape_str[:12] + "..."
        trainable = "Y" if param.requires_grad else "N"
        lines.append(
            f"| {name[:45]:<45} | {shape_str:<15} | {param.numel():<10,} | {trainable:<9} |"
        )
        
    if num_params > limit:
        lines.append(f"| ... [and {num_params - limit} more layers] {'':<46} |")
        
    lines.append("+" + "-"*80 + "+")
    
    # Calculate aggregation
    summary = get_trainable_parameters_summary(model)
    
    lines.append(f"| {'Trainable Parameters':<30} | {summary['trainable_parameters']:<45,} |")
    lines.append(f"| {'Frozen Parameters':<30} | {summary['frozen_parameters']:<45,} |")
    lines.append(f"| {'Total Parameters':<30} | {summary['total_parameters']:<45,} |")
    lines.append(f"| {'Trainable Percentage (%)':<30} | {summary['trainable_percentage']:<45.4f} |")
    lines.append("+" + "-"*80 + "+")
    
    summary_str = "\n".join(lines)
    print(summary_str)
    
    # Write summary report in logs
    logger.info("Model parameter summary printed.")
