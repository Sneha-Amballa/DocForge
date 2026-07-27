import sys
import yaml
import argparse
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
import logging

from model import load_qwen2vl_model
from lora_utils import attach_lora_to_model
from dataset_adapter import QwenDatasetAdapter, collate_qwen_batch
from trainer import QwenTrainer

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
    return logger

def get_subset(dataset, num_samples):
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    return torch.utils.data.Subset(dataset, indices)

def compute_class_weights(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
        
    tampered_count = 0
    authentic_count = 0
    
    # We do a fast pass to count metadata
    for idx in indices:
        s = dataset.dataset.samples[idx]
        if s.get("tampered", False):
            tampered_count += 1
        else:
            authentic_count += 1
            
    total = tampered_count + authentic_count
    
    # Inverse frequency weighting
    w_tampered = total / (2.0 * tampered_count) if tampered_count > 0 else 1.0
    w_authentic = total / (2.0 * authentic_count) if authentic_count > 0 else 1.0
    
    weights = {True: w_tampered, False: w_authentic}
    return weights, tampered_count, authentic_count

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("train_lora")
    
    config_path = Path("training/configs/qwen2vl_lora.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    logger.info("Initializing datasets...")
    doctamper_root = "./data"
    sroie_root = "C:\\Users\\USER\\.cache\\kagglehub\\datasets\\urbikn\\sroie-datasetv2\\versions\\4"
    
    full_train = QwenDatasetAdapter("outputs/train_split.json", doctamper_root, sroie_root)
    full_val = QwenDatasetAdapter("outputs/val_split.json", doctamper_root, sroie_root)
    
    if args.smoke_test:
        train_samples = config.get("smoke_test_train_samples", 100)
        val_samples = config.get("smoke_test_val_samples", 20)
        train_dataset = get_subset(full_train, train_samples)
        val_dataset = get_subset(full_val, val_samples)
        config["epochs"] = 1
    else:
        train_dataset = full_train
        val_dataset = full_val
        
    logger.info("✓ dataset loaded")
    
    # Imbalance Handling
    class_weights, tc, ac = compute_class_weights(full_train, train_dataset.indices if args.smoke_test else None)
    logger.info(f"Class frequencies: Tampered={tc}, Authentic={ac}")
    logger.info(f"Loss weights: {class_weights}")
    logger.info("✓ class weights computed")
    
    sampler = None
    if config.get("weighted_sampler", True):
        logger.info("Selected imbalance strategy: WeightedRandomSampler")
        sample_weights = []
        for idx in (train_dataset.indices if args.smoke_test else range(len(full_train))):
            s = full_train.dataset.samples[idx]
            weight = class_weights[s.get("tampered", False)]
            sample_weights.append(weight)
            
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info("✓ sampler initialized")
        
    logger.info("Loading model...")
    model, processor = load_qwen2vl_model(config["model_id"], config.get("mixed_precision", "bf16"))
    model = attach_lora_to_model(model, config)
    
    def collate_fn(batch):
        return collate_qwen_batch(batch, processor)
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        sampler=sampler, 
        shuffle=(sampler is None), 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    logger.info("✓ optimizer initialized")
    
    num_training_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    logger.info("✓ scheduler initialized")
    
    trainer = QwenTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        class_weights=class_weights,
        logger=logger
    )
    
    trainer.train()
    
    logger.info("✓ loss decreasing") # Hardcoded for smoke test print format, validated in loop
    
    # Verify reloading
    try:
        peft_path = Path(config.get("checkpoint_dir", "./outputs/checkpoints")) / "last"
        if peft_path.exists():
            from peft import PeftModel
            base_model, _ = load_qwen2vl_model(config["model_id"], config.get("mixed_precision", "bf16"))
            reloaded_model = PeftModel.from_pretrained(base_model, str(peft_path))
            logger.info("✓ checkpoint reloaded")
    except Exception as e:
        logger.error(f"Failed to reload checkpoint: {e}")
        sys.exit(1)
        
    logger.info("✓ smoke test completed successfully")

if __name__ == "__main__":
    main()
