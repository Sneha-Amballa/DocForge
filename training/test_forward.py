import sys
import yaml
import time
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader

from model import load_qwen2vl_model
from lora_utils import attach_lora_to_model
from dataset_adapter import QwenDatasetAdapter, collate_qwen_batch

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
    return logger

def test_forward_batch(model, processor, dataset, batch_size, logger):
    logger.info(f"\n--- Testing Batch Size: {batch_size} ---")
    
    # Custom collate with processor
    def collate_fn(batch):
        return collate_qwen_batch(batch, processor)
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    logger.info("✓ prompt generated")
    logger.info("✓ processor successful")
    
    # Move inputs to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    logger.info("✓ tensors created")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            logger.info(f"  {k} shape: {v.shape}")
            
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    model.train()
    
    start_time = time.time()
    with torch.no_grad(): # We do no_grad here just for memory safety during the pure forward test
        # Actually wait, the user wants "No optimizer. No backward pass."
        # We can run with torch.no_grad() to just verify the forward pass works and computes loss.
        # But let's let it build the graph to check real memory if we want, but no_grad is safer for simple verification
        pass
        
    # We will build the graph to accurately measure memory for a forward pass
    outputs = model(**inputs)
    
    end_time = time.time()
    
    logger.info("✓ forward pass successful")
    
    loss = outputs.loss
    logits = outputs.logits
    
    logger.info("✓ loss computed")
    logger.info(f"  Loss value: {loss.item():.4f}")
    
    logger.info("✓ logits generated")
    logger.info(f"  Logits shape: {logits.shape}")
    
    time_taken = end_time - start_time
    logger.info(f"  Time taken: {time_taken:.2f} seconds")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(f"  Peak GPU Memory usage: {peak_mem:.2f} GB")
    else:
        logger.info("  Memory usage: N/A (CPU)")

def main():
    logger = setup_logger("test_forward")
    
    config_path = Path("training/configs/qwen2vl_lora.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    model_id = config["model_id"]
    mixed_precision = config.get("mixed_precision", "bf16")
    
    logger.info("Loading model and processor...")
    model, processor = load_qwen2vl_model(model_id, mixed_precision, device_map="auto")
    model = attach_lora_to_model(model, config)
    
    logger.info("Initializing dataset adapter...")
    split_file = "outputs/val_split.json"
    doctamper_root = "./data"
    sroie_root = "C:\\Users\\USER\\.cache\\kagglehub\\datasets\\urbikn\\sroie-datasetv2\\versions\\4"
    
    dataset = QwenDatasetAdapter(split_file, doctamper_root, sroie_root)
    
    # Run tests for batch sizes 1, 2, 4
    for bsz in [1, 2, 4]:
        try:
            test_forward_batch(model, processor, dataset, bsz, logger)
        except Exception as e:
            logger.error(f"Forward pass failed on batch size {bsz}: {e}")
            sys.exit(1)
            
    logger.info("\n✓ batch test passed")
    logger.info("Phase 4B Verification Complete!")

if __name__ == "__main__":
    main()
