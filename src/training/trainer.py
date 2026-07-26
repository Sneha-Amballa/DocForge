import time
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DatasetConfig
from src.training.losses import DocForgeLossRegistry
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.logger import DocForgeTrainingLogger
from src.training.validator import validate_epoch
from src.training.utils import get_gpu_memory_usage, prepare_batch
from src.logger import get_logger

logger = get_logger("DocForge.Trainer")

class DocForgeTrainer:
    """Core trainer orchestrating the full LoRA fine-tuning workflow."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_registry: DocForgeLossRegistry,
        processor: Any,
        config: DatasetConfig
    ) -> None:
        """Initialize the trainer.

        Args:
            model: PEFT LoRA model.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            optimizer: Optimizer instance.
            scheduler: LR scheduler.
            loss_registry: Modular lossRegistry.
            processor: Qwen2VLDataProcessor instance.
            config: DatasetConfig instance containing training hyper-parameters.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_registry = loss_registry
        self.processor = processor
        self.config = config
        
        # Set target device
        # Resolve device from VLM Config (default to GPU if available)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        # Move model and loss registry
        self.model = self.model.to(self.device)
        self.loss_registry = self.loss_registry.to(self.device)
        
        # Resolve precision dtype
        self.precision_str = config.precision.lower().strip()
        if self.precision_str == "fp16":
            self.precision_dtype = torch.float16
        elif self.precision_str == "bf16":
            self.precision_dtype = torch.bfloat16
        else:
            self.precision_dtype = torch.float32
            
        # Autocast device type
        self.device_type = self.device.type
        self.autocast_enabled = (self.precision_dtype != torch.float32 and self.device_type in ["cuda", "cpu"])
        
        # Mixed precision GradScaler
        self.scaler = torch.amp.GradScaler(enabled=(self.precision_str == "fp16" and self.device_type == "cuda"))
        
        # Hyperparameters
        self.epochs = config.epochs
        self.grad_accum_steps = config.gradient_accumulation_steps
        self.grad_clip_norm = config.grad_clip_norm
        self.checkpoints_dir = config.checkpoints_dir
        self.checkpoint_interval = config.checkpoint_interval
        self.early_stopping_patience = config.early_stopping_patience
        self.dry_run = config.dry_run

        # Initialize logging backends
        log_dir = config.output_root / "logs"
        self.train_logger = DocForgeTrainingLogger(
            log_dir=str(log_dir),
            backends=config.logging_backends
        )

        # Internal state metrics
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(self) -> Dict[str, Any]:
        """Execute the complete training pipeline across multiple epochs.

        Returns:
            Dict[str, Any]: Summary of best training results.
        """
        logger.info("=" * 60)
        logger.info(f"STARTING LORA FINE-TUNING LOOP | DEVICE: {self.device}")
        logger.info(f"Precision: {self.precision_str} | Grad Accumulation Steps: {self.grad_accum_steps}")
        logger.info(f"Dry Run Mode: {self.dry_run}")
        logger.info("=" * 60)

        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            logger.info(f"\n--- Epoch {epoch} / {self.epochs} ---")
            
            # Run one training epoch
            train_metrics = self._train_epoch(epoch)
            
            # Run evaluation validation epoch
            val_metrics = validate_epoch(
                model=self.model,
                val_loader=self.val_loader,
                processor=self.processor,
                loss_registry=self.loss_registry,
                device=self.device,
                precision_dtype=self.precision_dtype
            )
            
            epoch_duration = time.time() - epoch_start
            
            # Combine metrics
            epoch_metrics = {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_duration_sec": epoch_duration
            }
            
            # Add loss components to logs
            for k, v in train_metrics.items():
                if k != "loss":
                    epoch_metrics[f"train_{k}"] = v
            for k, v in val_metrics.items():
                if k != "loss":
                    epoch_metrics[f"val_{k}"] = v
                    
            # Profile memory
            epoch_metrics["gpu_mem_gb"] = 0.0
            if torch.cuda.is_available():
                epoch_metrics["gpu_mem_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
                
            # Log epoch metrics to TensorBoard and Terminal
            self.train_logger.log_epoch(epoch, epoch_metrics)
            
            # Save checkpoint conditions
            val_loss = val_metrics["loss"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(f"New best validation loss reached: {self.best_val_loss:.5f}")
            else:
                self.patience_counter += 1
                
            if epoch % self.checkpoint_interval == 0 or is_best:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    checkpoints_dir=self.checkpoints_dir,
                    is_best=is_best
                )
                
            # Early stopping check
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered! No improvement in validation loss for {self.patience_counter} epochs.")
                break
                
        total_time = time.time() - start_time
        logger.info(f"Fine-tuning complete in {total_time:.2f}s. Best Val Loss: {self.best_val_loss:.5f}")
        
        self.train_logger.close()
        
        return {
            "best_val_loss": self.best_val_loss,
            "total_time_sec": total_time,
            "completed_epochs": epoch
        }

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch loop over the train dataset split."""
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        loss_components: Dict[str, float] = {}
        batch_start_time = time.time()
        
        num_batches = len(self.train_loader)
        steps_per_epoch = num_batches
        
        if self.dry_run:
            # Constrain to 5 steps per epoch in dry run mode
            steps_per_epoch = min(5, num_batches)
            
        logger.info(f"Training on {steps_per_epoch} batches...")
        
        for step, batch in enumerate(self.train_loader):
            if step >= steps_per_epoch:
                break
                
            # Process raw batch into model-ready tensors on target device
            batch = prepare_batch(batch, self.processor, self.device)
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]
            
            # Forward pass under autocast context
            with torch.amp.autocast(device_type=self.device_type, dtype=self.precision_dtype, enabled=self.autocast_enabled):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw
                )
                # Compute modular loss
                loss, loss_dict = self.loss_registry(outputs, batch)
                
            # Check for NaN / Infinity
            if not torch.isfinite(loss):
                logger.warning(f"NaN/Inf loss encountered at step {step}: {loss.item()}. Skipping update.")
                self.optimizer.zero_grad()
                continue
                
            # Normalize loss to account for gradient accumulation
            loss = loss / self.grad_accum_steps
            
            # Backward pass (using scaler if fp16)
            self.scaler.scale(loss).backward()
            
            # Optimizer step
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == steps_per_epoch:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                
                # Check for exploding gradients
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if not torch.isfinite(total_norm):
                    logger.warning(f"Exploding gradient norm detected ({total_norm.item()}). Skipping step.")
                    self.optimizer.zero_grad()
                    continue
                    
                # Optimizer step (using scaler)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.scheduler:
                    self.scheduler.step()
                    
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Step logging
                step_metrics = {
                    "loss": loss.item() * self.grad_accum_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "grad_norm": total_norm.item()
                }
                for k, v in loss_dict.items():
                    step_metrics[k] = v
                    
                self.train_logger.log_step(self.global_step, step_metrics)

            total_loss += loss.item() * self.grad_accum_steps
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0.0) + v
                
            # Log step progress bar every 10 steps (or every step if dry run)
            if step % (1 if self.dry_run else 10) == 0:
                elapsed = time.time() - batch_start_time
                steps_remaining = steps_per_epoch - step - 1
                avg_step_time = elapsed / (step + 1)
                eta = steps_remaining * avg_step_time
                
                # Format memory
                mem_str = get_gpu_memory_usage()
                
                logger.info(
                    f"Step {step}/{steps_per_epoch} | Loss: {loss.item() * self.grad_accum_steps:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s | "
                    f"ETA: {eta:.1f}s | GPU: {mem_str}"
                )
                
        # Average metrics
        train_metrics = {"loss": total_loss / steps_per_epoch}
        for k, v in loss_components.items():
            train_metrics[k] = v / steps_per_epoch
            
        return train_metrics
