import os
import sys
import json
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

try:
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Warning: torch is not installed.")
    Dataset = object

sys.path.insert(0, str(Path("eval").absolute()))
sys.path.insert(0, str(Path("data").absolute()))

from load_doctamper import LMDBDataset
from load_sroie2019 import SROIEDataset
from dataset_utils import setup_logger

class UnifiedDocForgeDataset(Dataset):
    _doctamper_loaders = {}
    _sroie_loader = None
    
    def __init__(self, split_file: str, doctamper_root: str, sroie_root: str):
        self.split_file = Path(split_file)
        with open(self.split_file, "r") as f:
            self.samples = json.load(f)
            
        self.sroie_root = sroie_root
        
    def _get_doctamper(self, path):
        if path not in self.__class__._doctamper_loaders:
            self.__class__._doctamper_loaders[path] = LMDBDataset(path)
        return self.__class__._doctamper_loaders[path]
        
    def _get_sroie(self):
        if self.__class__._sroie_loader is None:
            self.__class__._sroie_loader = SROIEDataset(self.sroie_root)
        return self.__class__._sroie_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        source = sample_meta["source"]
        dataset_idx = sample_meta["index"]
        
        if source == "doctamper":
            loader = self._get_doctamper(sample_meta["lmdb_path"])
            item = loader[dataset_idx]
        else:
            loader = self._get_sroie()
            item = loader[dataset_idx]
            
        return item

def check_sample(item):
    """Worker function for dataset verification"""
    warnings = []
    
    # Check image
    if "image_bytes" in item:
        img_array = np.frombuffer(item["image_bytes"], dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(item["image_path"])
        
    if img is None:
        warnings.append("Unreadable image")
        return {"valid": False, "warnings": warnings}
        
    h, w = img.shape[:2]
    
    # Check mask
    mask = item.get("mask_array")
    if item["tampered"]:
        if mask is None:
            warnings.append("Tampered image missing mask")
        else:
            if cv2.countNonZero(mask) == 0:
                warnings.append("Empty mask for tampered image")
                
    # Check bboxes
    bboxes = item["bounding_boxes"]
    for box in bboxes:
        if len(box) != 4:
            warnings.append("Invalid bounding box format")
            continue
            
        x_min, y_min, x_max, y_max = box
        if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
            warnings.append("Bounding box outside image")
            
        area = (x_max - x_min) * (y_max - y_min)
        if area < 10:
            warnings.append("Very small bounding box")
        if area > (h * w * 0.95):
            warnings.append("Very large bounding box")
            
    return {"valid": True, "warnings": warnings, "tampered": item["tampered"]}

def visualize_sample(item, out_path):
    if "image_bytes" in item:
        img_array = np.frombuffer(item["image_bytes"], dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(item["image_path"])
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((224, 224, 3), dtype=np.uint8)
    
    mask = item.get("mask_array")
    if mask is None:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
    img_with_boxes = img.copy()
    for bbox in item.get("bounding_boxes", []):
        if len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")
    
    axes[2].imshow(img_with_boxes)
    title = f"Tampered: {item['tampered']} | Forgery: {item.get('forgery_type')}"
    axes[2].set_title(title)
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--doctamper_root", type=str, default="./data")
    parser.add_argument("--sroie_root", type=str, default="C:\\Users\\USER\\.cache\\kagglehub\\datasets\\urbikn\\sroie-datasetv2\\versions\\4")
    parser.add_argument("--fast_mode", action="store_true", help="Skip full integrity check (for testing)")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("verify_dataset")
    out_dir = Path(args.out_dir)
    
    train_split_path = out_dir / "train_split.json"
    val_split_path = out_dir / "val_split.json"
    
    if not train_split_path.exists() or not val_split_path.exists():
        logger.error("Split files not found! Run prepare_splits.py first.")
        return
        
    doctamper_root = args.doctamper_root
    sroie_root = args.sroie_root
    
    split_config_path = out_dir / "split_config.json"
    if split_config_path.exists():
        import json
        with open(split_config_path, "r") as f:
            split_config = json.load(f)
        if doctamper_root == "./data":
            doctamper_root = split_config.get("doctamper_root", doctamper_root)
        if sroie_root == "C:\\Users\\USER\\.cache\\kagglehub\\datasets\\urbikn\\sroie-datasetv2\\versions\\4":
            sroie_root = split_config.get("sroie_root", sroie_root)
            
    logger.info("Initializing datasets...")
    train_dataset = UnifiedDocForgeDataset(str(train_split_path), doctamper_root, sroie_root)
    val_dataset = UnifiedDocForgeDataset(str(val_split_path), doctamper_root, sroie_root)
    
    logger.info(f"Train Dataset size: {len(train_dataset)}")
    logger.info(f"Validation Dataset size: {len(val_dataset)}")
    
    # Overlap Check
    train_ids = set([f"{s['source']}_{s['index']}" for s in train_dataset.samples])
    val_ids = set([f"{s['source']}_{s['index']}" for s in val_dataset.samples])
    
    overlap = train_ids.intersection(val_ids)
    if overlap:
        logger.error(f"FOUND {len(overlap)} OVERLAPPING SAMPLES BETWEEN TRAIN AND VAL!")
    else:
        logger.info("No overlap detected between Train and Validation splits.")
        
    duplicate_train = len(train_dataset.samples) - len(train_ids)
    duplicate_val = len(val_dataset.samples) - len(val_ids)
    
    if duplicate_train > 0 or duplicate_val > 0:
        logger.warning(f"Duplicates detected! Train: {duplicate_train}, Val: {duplicate_val}")
        
    # Dataset Integrity Verification
    if not args.fast_mode:
        logger.info("Starting deep dataset integrity check... (This may take several minutes)")
        # We will iterate manually over validation to demonstrate the check, and randomly sample training
        # to save time. Wait, the prompt says "Verify every sample in both splits."
        # We will verify all using multiprocessing. But LMDB is tricky with multiprocessing if opened
        # in the parent. So we just do single thread. For time sake we check only a subset.
        # But wait, user says "Verify every sample in both splits." 
        # I will check all samples!
        
        all_warnings = []
        unreadable = 0
        
        for i in tqdm(range(len(val_dataset)), desc="Verifying Validation Set"):
            item = val_dataset[i]
            res = check_sample(item)
            if not res["valid"]: unreadable += 1
            if res["warnings"]: all_warnings.extend(res["warnings"])
            
        for i in tqdm(range(len(train_dataset)), desc="Verifying Train Set"):
            item = train_dataset[i]
            res = check_sample(item)
            if not res["valid"]: unreadable += 1
            if res["warnings"]: all_warnings.extend(res["warnings"])
            
        report_path = out_dir / "dataset_integrity_report.md"
        with open(report_path, "w") as f:
            f.write("# Dataset Integrity Report\n\n")
            f.write(f"- Overlapping samples: {len(overlap)}\n")
            f.write(f"- Duplicates in Train: {duplicate_train}\n")
            f.write(f"- Duplicates in Val: {duplicate_val}\n")
            f.write(f"- Unreadable Images: {unreadable}\n\n")
            f.write("## Quality Warnings\n")
            if all_warnings:
                from collections import Counter
                counts = Counter(all_warnings)
                for k, v in counts.items():
                    f.write(f"- {k}: {v}\n")
            else:
                f.write("- No quality issues detected.\n")
                
        logger.info(f"Integrity report saved to {report_path}")
    else:
        logger.info("Fast mode enabled. Skipping deep integrity check.")
        report_path = out_dir / "dataset_integrity_report.md"
        with open(report_path, "w") as f:
            f.write("# Dataset Integrity Report\n\n")
            f.write(f"- Overlapping samples: {len(overlap)}\n")
            f.write("- Fast mode enabled: Deep integrity check was skipped.\n")

    # Visualization
    logger.info("Generating visualizations...")
    train_viz_dir = out_dir / "train_visualizations"
    val_viz_dir = out_dir / "validation_visualizations"
    train_viz_dir.mkdir(exist_ok=True)
    val_viz_dir.mkdir(exist_ok=True)
    
    train_indices = random.sample(range(len(train_dataset)), min(10, len(train_dataset)))
    for i, idx in enumerate(train_indices):
        visualize_sample(train_dataset[idx], str(train_viz_dir / f"train_viz_{i}.png"))
        
    val_indices = random.sample(range(len(val_dataset)), min(10, len(val_dataset)))
    for i, idx in enumerate(val_indices):
        visualize_sample(val_dataset[idx], str(val_viz_dir / f"val_viz_{i}.png"))
        
    logger.info("Visualizations generated.")
    
    # DataLoader Verification
    logger.info("Verifying PyTorch DataLoader compatibility...")
    try:
        def custom_collate(batch):
            # A simple collate function that doesn't stack images because sizes differ or it's just bytes
            # We just return the list of dicts.
            return batch
            
        dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
        
        batch = next(iter(dataloader))
        print("\n================ DATALOADER VERIFICATION ================")
        print(f"Batch Size: {len(batch)}")
        for i, item in enumerate(batch):
            print(f"--- Sample {i} ---")
            print(f"Tampered: {item['tampered']}")
            print(f"Bounding Boxes: {len(item.get('bounding_boxes', []))} boxes")
            print(f"Metadata: {item.get('metadata')}")
            
            if "image_bytes" in item:
                print("Image Data: Raw Bytes from LMDB")
            else:
                print(f"Image Data: Path -> {item.get('image_path')}")
        print("=========================================================\n")
        logger.info("DataLoader verification passed successfully!")
    except Exception as e:
        logger.error(f"DataLoader verification failed: {e}")

if __name__ == "__main__":
    main()
