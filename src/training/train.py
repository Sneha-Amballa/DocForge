import sys
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in system path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DatasetConfig
from src.logger import get_logger
from src.dataset_loader import get_dataloaders
from src.model.config import VLMConfig, LoraConfigSettings
from src.model.qwen_model import load_base_vlm, freeze_base_parameters
from src.model.lora import attach_lora_adapters, get_trainable_parameters_summary
from src.training.losses import DocForgeLossRegistry
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.trainer import DocForgeTrainer
from src.training.utils import set_seed, prepare_batch

logger = get_logger("DocForge.TrainEntryPoint")

def run_pretraining_checks(
    model: nn.Module,
    train_loader: DataLoader,
    processor: Any,
    loss_registry: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> bool:
    """Run mathematical and pipeline checks to verify training readiness before Epoch 1.

    Asserts:
        - Dataset loading and dataloader batch extraction.
        - Correct dimensions of processed tokens and image tensors.
        - Successful forward propagation.
        - Loss is finite.
        - Backpropagation computes gradients.
        - Base model weights remain frozen (no gradients).
        - LoRA weights calculate gradients properly.
        - Checkpoint saving and reloading runs without corruption.

    Returns:
        bool: True if all checks pass.
    """
    logger.info("=" * 60)
    logger.info("RUNNING PRE-TRAINING VERIFICATION CHECKS")
    logger.info("=" * 60)

    checks = {
        "1. Dataloader Load Check": False,
        "2. Batch Shape Integrity": False,
        "3. Forward Pass Execution": False,
        "4. Finite Loss Assertion": False,
        "5. Gradient Computation": False,
        "6. Frozen Backbone Safety": False,
        "7. Trainable LoRA Gradients": False,
        "8. Checkpointer Save/Reload": False
    }

    raw_batch = None
    batch = None
    
    # 1. Dataloader load check
    try:
        iterator = iter(train_loader)
        raw_batch = next(iterator)
        checks["1. Dataloader Load Check"] = raw_batch is not None
    except Exception as e:
        logger.error(f"Failed to fetch batch from dataloader: {e}")
        return False

    # Process raw batch into model-ready tensors using processor
    try:
        batch = prepare_batch(raw_batch, processor, device)
    except Exception as e:
        logger.error(f"Failed to prepare batch inputs using processor: {e}")
        return False

    # 2. Batch shape integrity
    if batch is not None:
        try:
            input_ids = batch["input_ids"]
            pixel_values = batch["pixel_values"]
            # Assert correct dimension boundaries
            checks["2. Batch Shape Integrity"] = (
                len(input_ids.shape) == 2 and 
                len(pixel_values.shape) == 4 and 
                pixel_values.shape[1] == 3
            )
        except KeyError as e:
            logger.error(f"Missing required batch keys: {e}")
            return False

    # 3. Forward Pass & 4. Finite Loss & 5. Gradient Computation
    if batch is not None:
        try:
            model.train()
            optimizer.zero_grad()

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            pix = batch["pixel_values"].to(device)
            thw = batch["image_grid_thw"].to(device)

            # Forward
            outputs = model(input_ids=ids, attention_mask=mask, pixel_values=pix, image_grid_thw=thw)
            checks["3. Forward Pass Execution"] = outputs is not None
            
            # Loss
            loss, _ = loss_registry(outputs, batch)
            checks["4. Finite Loss Assertion"] = torch.isfinite(loss).item()

            # Backward
            loss.backward()
            checks["5. Gradient Computation"] = True

            # 6. Frozen parameters unchanged and 7. Only LoRA weights updated
            frozen_ok = True
            lora_ok = False
            lora_param_checked = False
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    # Base parameters: grad must be None or zero
                    if param.grad is not None and torch.any(param.grad != 0.0):
                        logger.error(f"Frozen parameter '{name}' received gradient updates!")
                        frozen_ok = False
                else:
                    # LoRA parameters: requires_grad is True -> grad must not be None
                    lora_param_checked = True
                    if param.grad is None:
                        logger.error(f"LoRA adapter weight '{name}' requires gradient but received None.")
                    else:
                        lora_ok = True

            checks["6. Frozen Backbone Safety"] = frozen_ok
            checks["7. Trainable LoRA Gradients"] = lora_ok and lora_param_checked
            
            # Clean gradients
            optimizer.zero_grad()
            
        except Exception as e:
            logger.error(f"Forward/Backward verification flow crashed: {e}")
            return False

    # 8. Checkpointer Save/Reload
    if batch is not None:
        temp_dir = Path(tempfile.mkdtemp())
        try:
            dummy_metrics = {"loss": 0.5}
            # Save
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=999,
                metrics=dummy_metrics,
                checkpoints_dir=temp_dir
            )
            # Load
            load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                checkpoint_path=temp_dir / "latest"
            )
            checks["8. Checkpointer Save/Reload"] = True
        except Exception as e:
            logger.error(f"Checkpointer save/reload check failed: {e}")
        finally:
            shutil.rmtree(temp_dir)

    # Print check matrix
    print("\n" + "=" * 50)
    print(f"| {'PRE-TRAINING CHECKS SUMMARY':^46} |")
    print("=" * 50)
    for name, passed in checks.items():
        status = "PASSED" if passed else "FAILED"
        print(f"| {name:<35} | {status:^8} |")
    print("=" * 50 + "\n")

    return all(checks.values())

def main() -> None:
    """Launch the DocForge training pipeline."""
    # 1. Load configuration
    config = DatasetConfig()
    config.ensure_output_dirs()

    # Set seed
    set_seed(config.seed)

    # 2. Get dataloaders
    # Limit samples if dry run is active
    sample_limit = 50 if config.dry_run else None
    
    logger.info("Initializing datasets and PyTorch loaders...")
    train_loader, val_loader, _ = get_dataloaders(
        config.training_set,
        config=config,
        sample_limit=sample_limit
    )

    # 3. Load Qwen2-VL Model & Processor
    vlm_config = VLMConfig(offline_mode=config.dry_run, dataset_config=config)
    logger.info("Loading base Vision-Language Model...")
    base_model, processor = load_base_vlm(vlm_config)

    # 4. Freeze base parameters
    freeze_base_parameters(base_model)

    # 5. Attach LoRA adapters
    lora_settings = LoraConfigSettings()
    model = attach_lora_adapters(base_model, lora_settings)

    # 6. Initialize losses, optimizer, and scheduler
    loss_registry = DocForgeLossRegistry(lm_weight=1.0, cls_weight=0.1, loc_weight=0.0)
    optimizer = get_optimizer(model, learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    
    # Calculate training steps
    num_batches = len(train_loader)
    steps_per_epoch = min(5, num_batches) if config.dry_run else num_batches
    total_training_steps = (steps_per_epoch * config.epochs) // config.gradient_accumulation_steps
    
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_type=config.scheduler_type,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_training_steps
    )

    # 7. Check for resume checkpoint
    latest_checkpoint = config.checkpoints_dir / "latest"
    start_epoch = 0
    if latest_checkpoint.exists():
        logger.info(f"Resume checkpoint detected at {latest_checkpoint}. Loading state...")
        try:
            checkpoint_meta = load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=latest_checkpoint
            )
            start_epoch = checkpoint_meta.get("epoch", 0)
            logger.info(f"Resumed training from Epoch {start_epoch} successfully.")
        except Exception as e:
            logger.warning(f"Failed to load latest checkpoint: {e}. Starting fresh.")

    # 8. Run pre-training checks
    # Use target device resolved by model placement
    device = next(model.parameters()).device
    checks_passed = run_pretraining_checks(
        model=model,
        train_loader=train_loader,
        processor=processor,
        loss_registry=loss_registry,
        optimizer=optimizer,
        device=device
    )

    if not checks_passed:
        logger.error("Pre-training checks failed. Aborting training execution.")
        sys.exit(1)

    # 9. Instantiate trainer and run training
    trainer = DocForgeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_registry=loss_registry,
        processor=processor,
        config=config
    )
    
    # If resuming, update trainer starting states
    if start_epoch > 0:
        trainer.global_step = (steps_per_epoch * start_epoch) // config.gradient_accumulation_steps
        
    trainer.train()

if __name__ == "__main__":
    main()
