import torch
from typing import List, Union, Optional
from src.config import DatasetConfig

class VLMConfig:
    """Configurations for loading and operating the Qwen2-VL model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        torch_dtype: str = "float32",
        use_flash_attn: bool = False,
        offline_mode: bool = False,
        dataset_config: Optional[DatasetConfig] = None
    ) -> None:
        """Initialize the model configuration.

        Args:
            model_name: Hugging Face model identifier.
            device: Target execution device ('cuda', 'cpu', 'mps'). Defaults to auto-detection.
            torch_dtype: Precision string ('float32', 'float16', 'bfloat16').
            use_flash_attn: Whether to activate flash attention.
            offline_mode: Force offline mock loading (useful for dry runs and fast local tests).
            dataset_config: Parent DatasetConfig instance.
        """
        self.model_name = model_name
        self.use_flash_attn = use_flash_attn
        self.offline_mode = offline_mode
        self.dataset_config = dataset_config or DatasetConfig()

        # Auto-detect device
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Resolve PyTorch dtype
        self.torch_dtype = self._resolve_dtype(torch_dtype)

    def _resolve_dtype(self, dtype_str: str) -> torch.dtype:
        """Map dtype strings to PyTorch dtypes."""
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }
        key = dtype_str.lower().strip()
        if key not in mapping:
            raise ValueError(f"Unknown torch data type: {dtype_str}. Choose from: float32, float16, bfloat16.")
        return mapping[key]

    def __repr__(self) -> str:
        return (
            f"VLMConfig(\n"
            f"  model_name={self.model_name},\n"
            f"  device={self.device},\n"
            f"  torch_dtype={self.torch_dtype},\n"
            f"  use_flash_attn={self.use_flash_attn},\n"
            f"  offline_mode={self.offline_mode}\n"
            f")"
        )


class LoraConfigSettings:
    """Configurations for PEFT/LoRA adapter layers."""

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        task_type: str = "CAUSAL_LM"
    ) -> None:
        """Initialize LoRA parameters.

        Args:
            r: LoRA Rank.
            lora_alpha: Scaling factor (alpha).
            lora_dropout: Dropout probability.
            target_modules: Specific modules to inject adapters. Defaults to standard Qwen attention projection and MLP blocks.
            bias: Bias setting ('none', 'all', 'lora_only').
            task_type: Model task type (defaults to causal language modeling).
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type
        
        # Default Qwen2-VL attention target layers
        self.target_modules = target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]

    def __repr__(self) -> str:
        return (
            f"LoraConfigSettings(\n"
            f"  r={self.r},\n"
            f"  lora_alpha={self.lora_alpha},\n"
            f"  lora_dropout={self.lora_dropout},\n"
            f"  bias={self.bias},\n"
            f"  target_modules={self.target_modules}\n"
            f")"
        )
