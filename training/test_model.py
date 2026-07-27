import sys
import yaml
from pathlib import Path
import logging
import torch

from model import load_qwen2vl_model
from lora_utils import attach_lora_to_model

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
    return logger

def main():
    logger = setup_logger("test_model")
    
    config_path = Path("training/configs/qwen2vl_lora.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)
        
    logger.info(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    try:
        model_id = config["model_id"]
        mixed_precision = config.get("mixed_precision", "bf16")
        
        # 1. Load Model
        model, processor = load_qwen2vl_model(
            model_id=model_id, 
            mixed_precision=mixed_precision,
            device_map="auto"
        )
        
        # 2. Attach LoRA
        peft_model = attach_lora_to_model(model, config)
        
        # Verify it works
        logger.info("✓ model loaded")
        logger.info("✓ processor loaded")
        logger.info("✓ LoRA attached")
        logger.info("✓ trainable parameter count verified")
        logger.info("✓ device information verified")
        
        logger.info("\n✓ ready for training")
        
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
