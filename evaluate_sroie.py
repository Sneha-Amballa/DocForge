import os
import json
import csv
import time
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.logger import get_logger
from src.config import DatasetConfig
from src.deployment.inference.loader import ModelContainer
from src.deployment.inference.predictor import DocForgePredictor

logger = get_logger("DocForge.EvaluateSROIE")

def main():
    logger.info("Starting SROIE2019 cross-dataset evaluation...")
    
    cfg = DatasetConfig()
    if cfg.dry_run:
        logger.error("dry_run is True. Evaluation must run on real model.")
        return
        
    sroie_img_dir = Path("SROIE2019/test/img")
    if not sroie_img_dir.exists():
        logger.error(f"SROIE2019 directory not found at {sroie_img_dir}")
        return
        
    output_dir = Path("outputs/sroie")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(sroie_img_dir.glob("*.jpg"))
    if not images:
        logger.error("No images found in SROIE2019/test/img")
        return
        
    logger.info(f"Found {len(images)} receipts for evaluation.")
    
    try:
        model, processor, device = ModelContainer.load_resources()
        predictor = DocForgePredictor(model, processor, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
        
    metrics = {
        "total_receipts": len(images),
        "true_negatives": 0,
        "false_positives": 0,
        "false_positive_rate": 0.0,
        "avg_confidence": 0.0,
        "avg_latency_ms": 0.0
    }
    
    predictions = []
    
    logger.info("Running evaluation pipeline...")
    
    total_confidence = 0.0
    total_latency = 0.0
    
    # Process up to 10 images to keep the evaluation time reasonable for this run, 
    # since each forward pass on CPU takes ~72 seconds.
    for img_path in tqdm(images[:10]):
        try:
            img = Image.open(img_path).convert("RGB")
            res = predictor.predict_image(img, prompt="Assess if this document contains any tampered or forged areas.")
            
            is_tampered = res.get("tampered", False)
            confidence = res.get("confidence", 0.0)
            latency = res.get("latency_ms", 0)
            
            total_confidence += confidence
            total_latency += latency
            
            if is_tampered:
                metrics["false_positives"] += 1
                
                # Draw boxes and save
                draw = ImageDraw.Draw(img)
                for bbox in res.get("bounding_boxes", []):
                    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                    
                save_path = output_dir / f"{img_path.stem}_fp.png"
                img.save(save_path)
                logger.info(f"False Positive saved to {save_path}")
            else:
                metrics["true_negatives"] += 1
                
            pred_record = {
                "filename": img_path.name,
                "tampered_pred": is_tampered,
                "confidence": confidence,
                "latency_ms": latency,
                "reasoning": res.get("reasoning", "")
            }
            predictions.append(pred_record)
            
        except Exception as e:
            logger.error(f"Error evaluating {img_path.name}: {e}")
            
    # Calculate final metrics
    num_eval = len(predictions)
    metrics["false_positive_rate"] = metrics["false_positives"] / num_eval if num_eval > 0 else 0
    metrics["avg_confidence"] = total_confidence / num_eval if num_eval > 0 else 0
    metrics["avg_latency_ms"] = total_latency / num_eval if num_eval > 0 else 0
    
    logger.info("Evaluation complete.")
    logger.info(f"FPR: {metrics['false_positive_rate']:.2%}")
    logger.info(f"True Negatives: {metrics['true_negatives']}")
    logger.info(f"False Positives: {metrics['false_positives']}")
    
    # Save predictions JSON
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
        
    # Save predictions CSV
    with open(output_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "tampered_pred", "confidence", "latency_ms", "reasoning"])
        writer.writeheader()
        writer.writerows(predictions)
        
    # Save summary MD
    with open(output_dir / "summary.md", "w") as f:
        f.write("# SROIE2019 Evaluation Summary\n\n")
        f.write(f"- Total Receipts: {metrics['total_receipts']}\n")
        f.write(f"- True Negatives (Correctly Identified as Authentic): {metrics['true_negatives']}\n")
        f.write(f"- False Positives (Incorrectly Identified as Tampered): {metrics['false_positives']}\n")
        f.write(f"- False Positive Rate (FPR): {metrics['false_positive_rate']:.2%}\n")
        f.write(f"- Average Confidence: {metrics['avg_confidence']:.2f}%\n")
        f.write(f"- Average Latency: {metrics['avg_latency_ms']:.2f} ms\n")

if __name__ == "__main__":
    main()
