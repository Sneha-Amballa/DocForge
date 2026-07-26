from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from src.logger import get_logger

logger = get_logger("DocForge.TrainingLogger")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class DocForgeTrainingLogger:
    """Consolidated logging manager supporting Terminal, TensorBoard, and Weights & Biases."""

    def __init__(
        self,
        log_dir: str,
        backends: List[str]
    ) -> None:
        """Initialize loggers.

        Args:
            log_dir: Path to directory for logging events.
            backends: List of backends to initialize ('terminal', 'tensorboard', 'wandb').
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backends = [b.lower().strip() for b in backends]
        
        self.tb_writer: Optional[Any] = None
        self.wandb_active = False

        # Initialize TensorBoard
        if "tensorboard" in self.backends:
            if TENSORBOARD_AVAILABLE:
                try:
                    self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
                    logger.info(f"TensorBoard SummaryWriter initialized at {self.log_dir}")
                except Exception as e:
                    logger.error(f"Failed to initialize TensorBoard logger: {e}")
            else:
                logger.warning("TensorBoard not available. Skipping TensorBoard logging initialization.")

        # Initialize Weights & Biases
        if "wandb" in self.backends:
            try:
                import wandb
                # Initialize wandb run
                wandb.init(
                    project="DocForge",
                    dir=str(self.log_dir),
                    config={}
                )
                self.wandb_active = True
                logger.info("Weights & Biases logger initialized successfully.")
            except ImportError:
                logger.warning("Weights & Biases not installed. Skipping 'wandb' backend.")
            except Exception as e:
                logger.error(f"Failed to initialize W&B run: {e}")

    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Write metric scalars logged at a specific batch/optimization step.

        Args:
            step: Current training step.
            metrics: Metric name to float value mapping.
        """
        # TensorBoard step logging
        if self.tb_writer:
            for name, val in metrics.items():
                self.tb_writer.add_scalar(f"step/{name}", val, step)
                
        # Wandb step logging
        if self.wandb_active:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Write aggregated epoch-level metric values.

        Args:
            epoch: Current epoch index.
            metrics: Aggregated metrics name to float mapping.
        """
        # TensorBoard epoch logging
        if self.tb_writer:
            for name, val in metrics.items():
                self.tb_writer.add_scalar(f"epoch/{name}", val, epoch)
                
        # Terminal printing
        if "terminal" in self.backends:
            metrics_fmt = " | ".join([f"{k}: {v:.5f}" for k, v in metrics.items()])
            logger.info(f"Epoch {epoch} Summary: {metrics_fmt}")

        if self.wandb_active:
            try:
                import wandb
                # log with prefix
                epoch_metrics = {f"epoch_{k}": v for k, v in metrics.items()}
                epoch_metrics["epoch"] = epoch
                wandb.log(epoch_metrics)
            except Exception:
                pass

    def close(self) -> None:
        """safely terminate writers and finish logging events."""
        if self.tb_writer:
            self.tb_writer.close()
            logger.info("TensorBoard SummaryWriter closed.")
        if self.wandb_active:
            try:
                import wandb
                wandb.finish()
                logger.info("Weights & Biases logging session finished.")
            except Exception:
                pass
