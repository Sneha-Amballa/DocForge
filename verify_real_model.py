import sys
import os
import torch
import time
from PIL import Image
from src.logger import get_logger
from src.config import DatasetConfig
from src.deployment.inference.loader import ModelContainer
from src.deployment.inference.predictor import DocForgePredictor

logger = get_logger("DocForge.Verify")

def main():
    logger.info("Starting REAL model verification checklist...")
    cfg = DatasetConfig()
    
    if cfg.dry_run:
        logger.error("FAIL: dry_run is still True in config. Cannot verify real model.")
        sys.exit(1)
        
    logger.info("PASS: dry_run is False.")
    
    start = time.time()
    try:
        model, processor, device = ModelContainer.load_resources()
        logger.info(f"PASS: Resources loaded successfully in {time.time() - start:.2f}s on {device}.")
    except Exception as e:
        logger.error(f"FAIL: Error loading model/processor: {e}")
        sys.exit(1)
        
    # Check trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    if trainable > 0:
        logger.error(f"FAIL: Found {trainable} trainable parameters. The model should be completely frozen for inference!")
        sys.exit(1)
        
    logger.info(f"PASS: Model is fully frozen (0 trainable, {frozen} frozen parameters).")
    
    # Test Prediction
    logger.info("Testing sample forward pass...")
    predictor = DocForgePredictor(model, processor, device)
    
    if predictor.is_mock:
        logger.error("FAIL: Predictor loaded as mock despite dry_run=False.")
        sys.exit(1)
        
    logger.info("PASS: Predictor is using the REAL inference engine.")
    
    # Create dummy image
    img = Image.new('RGB', (800, 800), color=(255, 255, 255))
    try:
        res = predictor.predict_image(img, prompt="Assess if this document contains any tampered or forged areas.")
        logger.info(f"Prediction Output: {res}")
        logger.info(f"PASS: Forward pass succeeded with {res.get('latency_ms', 0)}ms latency and {res.get('confidence', 0):.2f}% confidence.")
    except Exception as e:
        logger.error(f"FAIL: Error during forward pass: {e}")
        sys.exit(1)
        
    logger.info("ALL VERIFICATION CHECKS PASSED.")

if __name__ == "__main__":
    main()
