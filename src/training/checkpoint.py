import json
import random
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.model.utils import save_lora_adapters, load_lora_adapters
from src.logger import get_logger

logger = get_logger("DocForge.Checkpoint")

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, Any],
    checkpoints_dir: Union[str, Path],
    is_best: bool = False,
    max_to_keep: int = 3
) -> Path:
    """Save training states and trainable model weights to a versioned checkpoint directory.

    Saves:
        - LoRA adapter parameters (.bin / .json).
        - Optimizer and Scheduler state dicts.
        - Training progress metrics, epoch, and random seed state.

    Args:
        model: PEFT-wrapped (or Mock) model.
        optimizer: PyTorch optimizer.
        scheduler: PyTorch learning rate scheduler.
        epoch: Current epoch index.
        metrics: Dictionary of metric values.
        checkpoints_dir: Root directory for checkpoints.
        is_best: True if this epoch has the best validation metric.
        max_to_keep: Max number of history checkpoints to keep.

    Returns:
        Path: Path to saved checkpoint folder.
    """
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Define folder paths
    epoch_dir = checkpoints_dir / f"checkpoint-epoch-{epoch}"
    latest_dir = checkpoints_dir / "latest"
    best_dir = checkpoints_dir / "best"
    
    # Save versioned checkpoint folder
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Save adapter weights (only trainable weights)
    save_lora_adapters(model, epoch_dir)
    
    # Compile training state metadata
    meta = {
        "epoch": epoch,
        "metrics": metrics,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "seed_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().tolist()
        }
    }
    
    # Write metadata dict
    torch.save(meta, epoch_dir / "training_state.pth")
    
    # Save metrics summaries JSON
    with open(epoch_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Checkpoint saved successfully to {epoch_dir}")

    # Symlink/copy to 'latest'
    _copy_dir_contents(epoch_dir, latest_dir)
    
    # Symlink/copy to 'best' if requested
    if is_best:
        logger.info(f"Epoch {epoch} is the new best! Copying checkpoint to 'best' folder...")
        _copy_dir_contents(epoch_dir, best_dir)

    # Manage max historical checkpoints to keep
    _cleanup_old_checkpoints(checkpoints_dir, max_to_keep)
    
    return epoch_dir

def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    checkpoint_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Load previously saved checkpoint states and weights.

    Restores:
        - LoRA adapter weights.
        - Optimizer learning parameters.
        - Scheduler learning rates.
        - Random seeds.

    Args:
        model: Base model to bind adapters to.
        optimizer: Optimizer instance to restore state.
        scheduler: Scheduler instance to restore state.
        checkpoint_path: Specific checkpoint directory or path. Defaults to loading 'latest'.

    Returns:
        Dict[str, Any]: Loaded checkpoint metadata.
    """
    if checkpoint_path is None:
        logger.error("No checkpoint path provided for loading.")
        raise ValueError("Must provide checkpoint_path to load_checkpoint.")
        
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint folder does not exist: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}...")
    
    # 1. Load adapter weights onto the base model
    loaded_model = load_lora_adapters(model, checkpoint_path)
    
    # 2. Load metadata states
    state_file = checkpoint_path / "training_state.pth"
    if not state_file.exists():
        logger.warning(f"No training metadata state found at {state_file}. Only weights loaded.")
        return {"epoch": 0, "metrics": {}}
        
    meta = torch.load(state_file, map_location="cpu", weights_only=False)
    
    # Restore optimizer state
    if optimizer is not None and "optimizer_state_dict" in meta:
        optimizer.load_state_dict(meta["optimizer_state_dict"])
        logger.info("Optimizer parameters state restored.")
        
    # Restore scheduler state
    if scheduler is not None and "scheduler_state_dict" in meta:
        scheduler.load_state_dict(meta["scheduler_state_dict"])
        logger.info("Scheduler decay steps state restored.")
        
    # Restore seeds
    if "seed_states" in meta:
        seeds = meta["seed_states"]
        random.setstate(seeds["python"])
        np.random.set_state(seeds["numpy"])
        torch.set_rng_state(torch.ByteTensor(seeds["torch"]))
        logger.info("Deterministic random seeds state restored.")
        
    return {
        "epoch": meta["epoch"],
        "metrics": meta.get("metrics", {})
    }


def _copy_dir_contents(src: Path, dst: Path) -> None:
    """Safely overwrite directory contents."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
        elif item.is_dir():
            shutil.copytree(item, dst / item.name)

def _cleanup_old_checkpoints(root: Path, max_to_keep: int) -> None:
    """Remove older epoch checkpoint directories exceeding max count."""
    epoch_dirs = sorted(
        [d for d in root.glob("checkpoint-epoch-*") if d.is_dir()],
        key=lambda d: int(d.name.split("-")[-1])
    )
    
    if len(epoch_dirs) > max_to_keep:
        to_remove = epoch_dirs[:-max_to_keep]
        for d in to_remove:
            logger.info(f"Removing old checkpoint folder: {d.name}")
            shutil.rmtree(d)
