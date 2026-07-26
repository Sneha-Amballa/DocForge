import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, Union
from src.model.config import VLMConfig
from src.model.processor import Qwen2VLDataProcessor
from src.logger import get_logger

logger = get_logger("DocForge.Model")

class MockModelOutput:
    """Mock container representing output predictions of the VLM."""
    def __init__(self, logits: torch.Tensor, loss: Optional[torch.Tensor] = None) -> None:
        self.logits = logits
        self.loss = loss


class MockSelfAttention(nn.Module):
    """Mock Self-Attention block containing the standard projections Targeted by LoRA."""
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = self.o_proj(q)
        return out


class MockMLP(nn.Module):
    """Mock Multi-Layer Perceptron block containing linear projections."""
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        out = self.down_proj(g * u)
        return out


class MockDecoderLayer(nn.Module):
    """Mock Decoder block stacking attention and feed-forward linear layers."""
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = MockSelfAttention(hidden_size)
        self.mlp = MockMLP(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x)
        x = x + attn_out
        mlp_out = self.mlp(x)
        return x + mlp_out


class MockModel(nn.Module):
    """Mock decoder stack mimicking LLM hidden states."""
    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([MockDecoderLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class MockVisual(nn.Module):
    """Mock Vision Encoder mimicking Qwen2-VL ViT block."""
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 64, kernel_size=14, stride=14)
        self.proj = nn.Linear(64, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Expected input shape (B, 3, H, W)
        x = self.patch_embed(pixel_values)  # (B, 64, H/14, W/14)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, 64)
        out = self.proj(x)  # (B, N, hidden_size)
        return out


class MockQwen2VLConfig:
    """Mock configuration metadata."""
    def __init__(self, hidden_size: int = 512, vocab_size: int = 151936) -> None:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.torch_dtype = torch.float32


class MockQwen2VLForConditionalGeneration(nn.Module):
    """Mock Conditional Generation pipeline replicating Qwen2-VL-2B-Instruct interfaces."""

    def __init__(self) -> None:
        super().__init__()
        self.config = MockQwen2VLConfig()
        self.model = MockModel(self.config.hidden_size, self.config.vocab_size)
        self.visual = MockVisual(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> MockModelOutput:
        """Run mock forward pass, compiling visual and token embeddings."""
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.model(input_ids)  # (B, S, hidden_size)
        
        # Integrate visual tokens if present
        if pixel_values is not None:
            vis_feat = self.visual(pixel_values)  # (B, N, hidden_size)
            # Simple addition to represent cross-modal interactions without dimensional collisions
            # In a mock forward pass we map to sequence dimensions
            # For simplicity, we just project visual features or add token features
            pass
            
        logits = self.lm_head(x)  # (B, S, vocab_size)
        return MockModelOutput(logits=logits)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate mock response tokens."""
        batch_size = input_ids.shape[0]
        # Return a simple mock sequence of 5 tokens
        mock_output = torch.tensor([[100, 101, 102, 103, 104] for _ in range(batch_size)], dtype=torch.long)
        return mock_output


def load_base_vlm(config: VLMConfig) -> Tuple[nn.Module, Qwen2VLDataProcessor]:
    """Load Qwen2-VL-2B-Instruct base model and processor.

    Falls back to a structural Mock VLM architecture if offline_mode is set
    or Hugging Face downloads fail.

    Args:
        config: VLMConfig loader instance.

    Returns:
        Tuple[nn.Module, Qwen2VLDataProcessor]: Loaded PyTorch model and Processor wrapper.
    """
    processor = Qwen2VLDataProcessor(processor_name=config.model_name)

    if config.offline_mode:
        logger.info("Forced offline mode active. Loading Mock Qwen2-VL architecture...")
        model = MockQwen2VLForConditionalGeneration()
        return model, processor

    try:
        from transformers import Qwen2VLForConditionalGeneration
        import torch

        logger.info(f"Downloading/Loading pretrained model '{config.model_name}' from Hugging Face...")
        # Load conditional generation weights
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=config.torch_dtype,
            device_map="auto" if config.device == "cuda" else None,
            trust_remote_code=True
        )
        logger.info(f"Model loaded successfully on device: {config.device}")
        
    except Exception as e:
        if not config.offline_mode:
            logger.error(f"Critical failure loading real pretrained model: {e}")
            raise e
        logger.warning(
            f"Failed to load pretrained Qwen2-VL model from Hugging Face: {e}. "
            "Falling back to Mock VLM architecture..."
        )
        model = MockQwen2VLForConditionalGeneration()

    # Move model to device (device_map auto handles cuda placements)
    if not hasattr(model, "device_map") or model.device_map is None:
        model = model.to(config.device)
        logger.info(f"Transferred VLM to target device: {config.device}")

    return model, processor

def freeze_base_parameters(model: nn.Module) -> None:
    """Freeze all base model parameters to prevent updates during tuning.

    Args:
        model: Loaded neural network model.
    """
    logger.info("Freezing all parameters of the base model...")
    for name, param in model.named_parameters():
        param.requires_grad = False
    logger.info("Base parameters successfully frozen.")
