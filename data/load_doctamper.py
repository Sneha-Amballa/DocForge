import os
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import lmdb

try:
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from torch.utils.data import Dataset, ConcatDataset
except ImportError:
    print("Warning: torch is not installed. PyTorch Dataset functionality will be limited.")
    Dataset = object

from dataset_utils import (
    get_lmdb_envs,
    read_image_from_lmdb,
    extract_bboxes_from_mask,
    setup_logger
)

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path: str):
        """
        Unified loader for a single DocTamper LMDB dataset.
        """
        self.lmdb_path = Path(lmdb_path)
        self.env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False)
        
        with self.env.begin() as txn:
            num_samples_bytes = txn.get(b'num-samples')
            if num_samples_bytes:
                self.length = int(num_samples_bytes.decode('utf-8'))
            else:
                self.length = sum(1 for key, _ in txn.cursor() if key.startswith(b'image-'))
                
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # We need a new transaction for each thread
        with self.env.begin() as txn:
            img_key = f'image-{idx:09d}'.encode('utf-8')
            mask_key = f'label-{idx:09d}'.encode('utf-8')
            
            img_val = txn.get(img_key)
            mask = read_image_from_lmdb(txn, mask_key, grayscale=True)
            
            # Since the user requested just image_path, we must return bytes or image array
            # We'll save the path as the LMDB key for reference
            
            tampered = False
            bboxes = []
            forgery_type = "unknown"
            
            if mask is not None and cv2.countNonZero(mask) > 0:
                tampered = True
                bboxes = extract_bboxes_from_mask(mask)
                
            metadata = {
                "image_key": img_key.decode('utf-8'),
                "mask_key": mask_key.decode('utf-8') if txn.get(mask_key) else None,
                "split": self.lmdb_path.name,
                "index": idx
            }
            
            return {
                "image_bytes": img_val,
                "mask_array": mask,
                "tampered": tampered,
                "bounding_boxes": bboxes,
                "forgery_type": forgery_type,
                "metadata": metadata
            }

def visualize_sample(sample, out_path: str):
    """
    Visualizes original image, binary mask, and bounding boxes.
    Saves to out_path.
    """
    img_array = np.frombuffer(sample["image_bytes"], dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((224, 224, 3), dtype=np.uint8)
    
    mask = sample["mask_array"]
    if mask is None:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
    # Draw bboxes on image
    img_with_boxes = img.copy()
    for bbox in sample["bounding_boxes"]:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Binary Mask")
    axes[1].axis("off")
    
    axes[2].imshow(img_with_boxes)
    title = f"Tampered: {sample['tampered']}\nBoxes: {len(sample['bounding_boxes'])}"
    axes[2].set_title(title)
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Unified DocTamper LMDB Dataset Loader")
    parser.add_argument("--root", type=str, default="./data", help="Root directory containing LMDBs")
    parser.add_argument("--out_dir", type=str, default="./outputs/sample_visualizations", help="Output directory")
    args = parser.parse_args()
    
    logger = setup_logger("load_doctamper")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing dataset loader...")
    lmdb_dirs = get_lmdb_envs(args.root)
    
    if not lmdb_dirs:
        logger.warning("No LMDB samples found. Check your dataset root.")
        return
        
    datasets = [LMDBDataset(str(d)) for d in lmdb_dirs]
    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    
    logger.info(f"Loaded {len(full_dataset)} total samples across {len(datasets)} LMDB splits.")
        
    logger.info("Visualizing 10 random samples...")
    indices = random.sample(range(len(full_dataset)), min(10, len(full_dataset)))
    
    for i, idx in enumerate(indices):
        sample = full_dataset[idx]
        out_path = str(out_dir / f"viz_sample_{i}.png")
        visualize_sample(sample, out_path)
        logger.info(f"Saved visualization to {out_path}")
        
    logger.info("Visualization complete.")

if __name__ == "__main__":
    main()
