import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_torch_dtype(mixed_precision_str: str) -> torch.dtype:
    """Returns the appropriate torch dtype based on config string and hardware."""
    if mixed_precision_str == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def load_qwen2vl_model(model_id: str, mixed_precision: str = "bf16", device_map: str = "auto"):
    """
    Loads Qwen2-VL model and processor.
    Returns: (model, processor)
    """
    logger = setup_logger("model_loader")
    dtype = get_torch_dtype(mixed_precision)
    
    logger.info(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    logger.info(f"Loading model {model_id} with dtype={dtype} and device_map={device_map}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map
    )
    
    logger.info(f"✓ Model {model_id} successfully loaded.")
    logger.info(f"✓ Processor successfully loaded.")
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {total_params:,}")
    logger.info(f"Model Dtype: {model.dtype}")
    
    # Check device placement
    devices = set(p.device for p in model.parameters())
    logger.info(f"Model Devices: {devices}")
    
    return model, processor
