import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

from src.logger import get_logger
from src.config import DatasetConfig
from src.model.config import VLMConfig, LoraConfigSettings
from src.model.qwen_model import load_base_vlm
from src.model.utils import load_lora_adapters
from src.processor import Qwen2VLDataProcessor

logger = get_logger("DocForge.DeployLoader")

class ModelContainer:
    """Singleton cache container storing loaded model and processor instances."""

    _model: Optional[nn.Module] = None
    _processor: Optional[Qwen2VLDataProcessor] = None
    _device: Optional[torch.device] = None
    _config: Optional[DatasetConfig] = None
    _loading_time_ms: float = 0.0

    @classmethod
    def load_resources(cls, force_reload: bool = False) -> Tuple[nn.Module, Qwen2VLDataProcessor, torch.device]:
        """Load and cache base VLM and LoRA adapters.

        Args:
            force_reload: Re-reads checkpoints from disk even if already cached.

        Returns:
            Tuple[nn.Module, Qwen2VLDataProcessor, torch.device]: Model, Processor, and Device.
        """
        if not force_reload and cls._model is not None and cls._processor is not None:
            return cls._model, cls._processor, cls._device

        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        cpu_start = time.time() if start_time is None else None

        if start_time:
            start_time.record()

        logger.info("Initializing deployment resources...")
        cls._config = DatasetConfig()
        
        # Determine device
        if torch.cuda.is_available():
            cls._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            cls._device = torch.device("mps")
        else:
            cls._device = torch.device("cpu")
            
        logger.info(f"Target deployment execution device: {cls._device}")

        # Config structures
        vlm_config = VLMConfig(
            torch_dtype=cls._config.precision,
            device=str(cls._device),
            offline_mode=cls._config.dry_run
        )
        
        # Load VLM
        base_model, cls._processor = load_base_vlm(vlm_config)

        # Attach LoRA adapters
        checkpoints_dir = cls._config.checkpoints_dir
        latest_checkpoint = checkpoints_dir / "latest"
        if not latest_checkpoint.exists():
            latest_checkpoint = checkpoints_dir / "best"
            
        if not latest_checkpoint.exists():
            logger.warning(f"No trained LoRA adapters found at {checkpoints_dir}. Serving fresh base model.")
            cls._model = base_model
        else:
            logger.info(f"Loading trained LoRA checkpoint from {latest_checkpoint}...")
            try:
                cls._model = load_lora_adapters(base_model, latest_checkpoint)
            except Exception as e:
                logger.error(f"Error loading trained adapters: {e}. Falling back to base model.")
                cls._model = base_model

        cls._model = cls._model.to(cls._device)
        cls._model.eval()

        # Stop timer
        if start_time and end_time:
            end_time.record()
            torch.cuda.synchronize()
            cls._loading_time_ms = float(start_time.elapsed_time(end_time))
        else:
            cls._loading_time_ms = float((time.time() - cpu_start) * 1000.0)

        logger.info(f"Resources loaded successfully. Loading time: {cls._loading_time_ms:.2f} ms")
        return cls._model, cls._processor, cls._device

    @classmethod
    def get_loading_time_ms(cls) -> float:
        """Retrieve model startup load latency."""
        return cls._loading_time_ms

    @classmethod
    def get_metrics_summary(cls) -> Dict[str, Any]:
        """Compile deployment container status statistics."""
        return {
            "device": str(cls._device) if cls._device else "Uninitialized",
            "model_loaded": cls._model is not None,
            "processor_loaded": cls._processor is not None,
            "loading_time_ms": cls._loading_time_ms
        }
