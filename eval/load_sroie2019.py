import os
import argparse
import random
import json
import csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

try:
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from torch.utils.data import Dataset
except ImportError:
    print("Warning: torch is not installed. PyTorch Dataset functionality will be limited.")
    Dataset = object

from sroie_utils import (
    find_files_by_extensions,
    parse_sroie_ocr,
    match_images_and_annotations,
    setup_logger
)

class SROIEDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        
        img_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        ann_extensions = {".txt", ".json", ".csv"}
        
        self.image_files = find_files_by_extensions(root_dir, img_extensions)
        self.ann_files = find_files_by_extensions(root_dir, ann_extensions)
        self.matched = match_images_and_annotations(self.image_files, self.ann_files)
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        ann_path = self.matched.get(img_path)
        
        ocr_boxes = []
        ocr_text = []
        if ann_path and ann_path.exists():
            ocr_boxes, ocr_text = parse_sroie_ocr(ann_path)
            
        metadata = {
            "dataset": "SROIE2019",
            "image_filename": img_path.name,
            "ocr_boxes": ocr_boxes,
            "ocr_text": ocr_text
        }
        
        return {
            "image_path": str(img_path.absolute()),
            "mask_path": None,
            "tampered": False,
            "bounding_boxes": [],
            "forgery_type": None,
            "metadata": metadata
        }

def get_image_size(filepath: str) -> tuple:
    try:
        with Image.open(filepath) as img:
            return img.size
    except Exception:
        return (-1, -1)

def is_valid_image(filepath: str) -> bool:
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def visualize_sample(sample, out_path: str):
    img = cv2.imread(sample["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((224, 224, 3), dtype=np.uint8)
    
    img_with_boxes = img.copy()
    ocr_boxes = sample["metadata"]["ocr_boxes"]
    
    for bbox in ocr_boxes:
        if len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(img_with_boxes)
    title = f"OOD Sample - Tampered: {sample['tampered']}\nOCR Boxes: {len(ocr_boxes)}"
    axes[1].set_title(title)
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Load and Validate SROIE2019 Dataset")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    args = parser.parse_args()
    
    logger = setup_logger("load_sroie2019")
    root_path = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Inspecting SROIE2019 dataset at {root_path.absolute()}")
    
    img_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    ann_extensions = {".txt", ".json", ".csv"}
    
    image_files = find_files_by_extensions(args.root, img_extensions)
    ann_files = find_files_by_extensions(args.root, ann_extensions)
    
    # 1. Dataset Discovery
    train_folders = set()
    test_folders = set()
    image_folders = set()
    ann_folders = set()
    
    for f in image_files:
        parts = [p.lower() for p in f.parts]
        if 'train' in parts: train_folders.add(str(f.parent))
        if 'test' in parts: test_folders.add(str(f.parent))
        image_folders.add(str(f.parent))
        
    for f in ann_files:
        parts = [p.lower() for p in f.parts]
        if 'train' in parts: train_folders.add(str(f.parent))
        if 'test' in parts: test_folders.add(str(f.parent))
        ann_folders.add(str(f.parent))
        
    img_exts = {f.suffix.lower() for f in image_files}
    ann_exts = {f.suffix.lower() for f in ann_files}
    
    print("-" * 48)
    print("Dataset Summary:")
    print(f"Root: {root_path}")
    print(f"Train folders detected: {len(train_folders)}")
    print(f"Test folders detected: {len(test_folders)}")
    print(f"Image folders detected: {len(image_folders)}")
    print(f"OCR annotation folders detected: {len(ann_folders)}")
    print(f"Image extensions: {', '.join(img_exts)}")
    print(f"Annotation extensions: {', '.join(ann_exts)}")
    print("-" * 48)
    
    # 2. Dataset Validation
    matched = match_images_and_annotations(image_files, ann_files)
    
    missing_images = 0 # Handled inherently, we iterate through images
    missing_ocr = len(image_files) - len(matched)
    unreadable_images = 0
    duplicate_filenames = len(image_files) - len(set(f.name for f in image_files))
    
    resolutions = []
    ext_dist = defaultdict(int)
    
    logger.info("Validating dataset files...")
    for img_path in tqdm(image_files, desc="Validating Images"):
        ext_dist[img_path.suffix.lower()] += 1
        if not is_valid_image(str(img_path)):
            unreadable_images += 1
        else:
            size = get_image_size(str(img_path))
            if size != (-1, -1):
                resolutions.append(size)
                
    report_path = out_dir / "sroie_validation_report.md"
    with open(report_path, "w") as f:
        f.write("# SROIE2019 Validation Report\n\n")
        f.write(f"- Total Images Found: {len(image_files)}\n")
        f.write(f"- Total OCR Files Found: {len(ann_files)}\n")
        f.write(f"- Missing OCR Files: {missing_ocr}\n")
        f.write(f"- Unreadable Images: {unreadable_images}\n")
        f.write(f"- Duplicate Filenames: {duplicate_filenames}\n")
    logger.info(f"Saved validation report to {report_path}")
    
    # 3. Dataset Statistics
    if resolutions:
        avg_w = sum(w for w, h in resolutions) / len(resolutions)
        avg_h = sum(h for w, h in resolutions) / len(resolutions)
        min_res = min(resolutions)
        max_res = max(resolutions)
        
        stats = {
            "Total receipts": len(image_files),
            "Average resolution": f"{int(avg_w)}x{int(avg_h)}",
            "Minimum resolution": f"{min_res[0]}x{min_res[1]}",
            "Maximum resolution": f"{max_res[0]}x{max_res[1]}",
            "Image extensions": dict(ext_dist)
        }
        
        json_path = out_dir / "sroie_statistics.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=4)
            
        csv_path = out_dir / "sroie_statistics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(stats.keys())
            writer.writerow([str(v) for v in stats.values()])
            
        logger.info(f"Saved statistics to {json_path} and {csv_path}")

    # 4 & 5 & 7. Unified Dataset Loader
    logger.info("Initializing PyTorch Dataset...")
    dataset = SROIEDataset(str(root_path))
    logger.info(f"Loaded {len(dataset)} samples in unified format.")
    
    if len(dataset) > 0:
        print("\n--- Unified Format Example (Sample 0) ---")
        sample_example = dataset[0]
        # Truncate large lists for printing
        sample_example_print = sample_example.copy()
        if len(sample_example_print["metadata"]["ocr_boxes"]) > 2:
            sample_example_print["metadata"]["ocr_boxes"] = sample_example_print["metadata"]["ocr_boxes"][:2] + ["..."]
        if len(sample_example_print["metadata"]["ocr_text"]) > 2:
            sample_example_print["metadata"]["ocr_text"] = sample_example_print["metadata"]["ocr_text"][:2] + ["..."]
        print(json.dumps(sample_example_print, indent=4))
        
    # 6. Visualization
    viz_dir = out_dir / "sroie_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    if len(dataset) > 0:
        logger.info("Visualizing 10 random receipts...")
        indices = random.sample(range(len(dataset)), min(10, len(dataset)))
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            out_path = str(viz_dir / f"sroie_viz_{i}.png")
            visualize_sample(sample, out_path)
        logger.info(f"Saved visualizations to {viz_dir}")

if __name__ == "__main__":
    main()
