import random
import os
from pathlib import Path
from typing import Union, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from src.logger import get_logger

logger = get_logger("DocForge.ModelUtils")

def set_seed(seed: int = 42) -> None:
    """Set global seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: Seed value.
    """
    logger.info(f"Setting global random seed to {seed}...")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic ops
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Seeds set successfully.")

def save_lora_adapters(model: nn.Module, save_path: Union[str, Path]) -> None:
    """Save the trainable LoRA adapter weights to a local directory.

    Args:
        model: PEFT-wrapped (or Mock) model.
        save_path: Output directory path.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Check if model has standard PEFT save pretrained method
    if hasattr(model, "save_pretrained"):
        try:
            logger.info(f"Saving PEFT LoRA adapters to {save_path}...")
            model.save_pretrained(str(save_path))
            logger.info("PEFT adapters saved successfully.")
            return
        except Exception as e:
            logger.error(f"Failed to save PEFT adapters: {e}")
            
    # Fallback/Mock state saving
    logger.info(f"Saving mock adapter state dict to {save_path}...")
    state_dict = {}
    
    # Extract only trainable weights (LoRA weights)
    for name, param in model.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.data.cpu()
            
    state_path = save_path / "adapter_model.bin"
    torch.save(state_dict, state_path)
    
    # Save a dummy config.json for PEFT compatibility
    config_path = save_path / "adapter_config.json"
    import json
    dummy_config = {
        "r": getattr(getattr(model, "settings", None), "r", 8),
        "lora_alpha": getattr(getattr(model, "settings", None), "lora_alpha", 16),
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM"
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(dummy_config, f, indent=4)
        
    logger.info(f"Mock state dict successfully saved to {state_path}")

def load_lora_adapters(
    base_model: nn.Module,
    adapter_path: Union[str, Path]
) -> nn.Module:
    """Load previously saved LoRA adapter weights and bind them to the base model.

    Args:
        base_model: Pretrained base model.
        adapter_path: Directory path containing adapter weights.

    Returns:
        nn.Module: Base model wrapped with loaded adapters.
    """
    adapter_path = Path(adapter_path)
    
    # Check if peft from_pretrained is available
    try:
        from peft import PeftModel
        logger.info(f"Attempting to load PEFT adapters from {adapter_path}...")
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        logger.info("PEFT adapters loaded successfully.")
        return peft_model
    except Exception as e:
        from src.config import DatasetConfig
        cfg = DatasetConfig()
        if not cfg.dry_run:
            logger.error(f"Critical failure loading PEFT adapters: {e}")
            raise e
        logger.warning(f"Could not load adapters using PEFT: {e}. Falling back to Mock state loader.")

    # Mock loading fallback
    state_path = adapter_path / "adapter_model.bin"
    config_path = adapter_path / "adapter_config.json"
    
    if not state_path.exists():
        logger.error(f"Adapter state file not found: {state_path}")
        raise FileNotFoundError(f"Adapter file not found: {state_path}")
        
    logger.info(f"Loading mock adapter weights from {state_path}...")
    import json
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
        
    from src.model.config import LoraConfigSettings
    from src.model.lora import attach_lora_adapters
    
    settings = LoraConfigSettings(
        r=config_data.get("r", 8),
        lora_alpha=config_data.get("lora_alpha", 16)
    )
    
    # Attach adapters (creates structural param weights list) if not already wrapped
    if hasattr(base_model, "lora_adapters"):
        model = base_model
        logger.debug("Base model is already a MockPeftModel. Skipping adapter re-attachment.")
    else:
        model = attach_lora_adapters(base_model, settings)
    
    # Load state dict values
    state_dict = torch.load(state_path)
    
    # Load parameters
    # The loaded parameter names map to the wrapped model parameters
    # Using strict=False since we only save requires_grad=True parameters
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded weights with missing keys: {len(missing_keys)}, unexpected: {len(unexpected_keys)}")
    
    return model
