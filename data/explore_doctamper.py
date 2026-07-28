import os
import argparse
import random
import json
import csv
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
import lmdb

from dataset_utils import (
    get_lmdb_envs,
    read_image_from_lmdb,
    get_jpeg_size,
    get_png_size,
    setup_logger
)

def parse_args():
    parser = argparse.ArgumentParser(description="Explore DocTamper LMDB Dataset")
    parser.add_argument("--root", type=str, default="./data", help="Root directory of the dataset")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory for reports")
    return parser.parse_args()

def validate_lmdb(lmdb_path: Path):
    """Validates a single LMDB dataset and returns statistics."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    
    total_images = 0
    missing_masks = 0
    corrupted_images = 0
    corrupted_masks = 0
    size_mismatches = 0
    
    tampered_count = 0
    authentic_count = 0
    resolutions = []
    
    with env.begin() as txn:
        # Get total samples
        num_samples_bytes = txn.get(b'num-samples')
        if num_samples_bytes:
            total_images = int(num_samples_bytes.decode('utf-8'))
        else:
            # Manually count
            total_images = sum(1 for key, _ in txn.cursor() if key.startswith(b'image-'))
            
        print(f"Validating {lmdb_path.name} with {total_images} samples...")
        
        for i in tqdm(range(total_images), desc=f"Processing {lmdb_path.name}"):
            img_key = f'image-{i:09d}'.encode('utf-8')
            mask_key = f'label-{i:09d}'.encode('utf-8')
            
            img_val = txn.get(img_key)
            mask_val = txn.get(mask_key)
            
            img_size = (-1, -1)
            mask_size = (-1, -1)
            
            if img_val is None:
                corrupted_images += 1
                continue
            else:
                img_size = get_jpeg_size(img_val)
                if img_size == (-1, -1):
                    corrupted_images += 1
                else:
                    resolutions.append(img_size)
                    
            if mask_val is None:
                missing_masks += 1
                authentic_count += 1
            else:
                mask_size = get_png_size(mask_val)
                if mask_size == (-1, -1):
                    corrupted_masks += 1
                    authentic_count += 1
                else:
                    if img_size != mask_size:
                        size_mismatches += 1
                        
                    # Check tampering
                    mask = read_image_from_lmdb(txn, mask_key, grayscale=True)
                    if mask is not None and cv2.countNonZero(mask) > 0:
                        tampered_count += 1
                    else:
                        authentic_count += 1
                        
    return {
        "lmdb_name": lmdb_path.name,
        "total_images": total_images,
        "missing_masks": missing_masks,
        "corrupted_images": corrupted_images,
        "corrupted_masks": corrupted_masks,
        "size_mismatches": size_mismatches,
        "tampered": tampered_count,
        "authentic": authentic_count,
        "resolutions": resolutions
    }

def main():
    args = parse_args()
    logger = setup_logger("explore_doctamper")
    
    root_path = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exploring LMDB dataset at {root_path.absolute()}")
    
    lmdb_dirs = get_lmdb_envs(args.root)
    if not lmdb_dirs:
        logger.error("No LMDB datasets found! Are you sure data.mdb exists in subfolders?")
        return
        
    print("-" * 48)
    print("Dataset Root:")
    print(str(root_path.absolute()))
    
    splits = [d.name for d in lmdb_dirs]
    print("\nDetected LMDB Splits:")
    for s in splits:
        print(f" - {s}")
    print("-" * 48)
    
    # Randomly inspect one sample from the first LMDB
    print("\n--- Inspecting Random Sample from First LMDB ---")
    env = lmdb.open(str(lmdb_dirs[0]), readonly=True, lock=False)
    with env.begin() as txn:
        num_samples_bytes = txn.get(b'num-samples')
        if num_samples_bytes:
            total_samples = int(num_samples_bytes.decode('utf-8'))
            rand_idx = random.randint(0, total_samples - 1)
            
            img_key = f'image-{rand_idx:09d}'.encode('utf-8')
            mask_key = f'label-{rand_idx:09d}'.encode('utf-8')
            
            img = read_image_from_lmdb(txn, img_key)
            mask = read_image_from_lmdb(txn, mask_key, grayscale=True)
            
            print(f"Sample Index: {rand_idx}")
            print(f"Image key: {img_key.decode()} | size: {img.shape if img is not None else 'N/A'}")
            print(f"Mask key: {mask_key.decode()} | size: {mask.shape if mask is not None else 'N/A'}")
            if mask is not None:
                print(f"Unique mask values: {np.unique(mask).tolist()}")
            print("-" * 30)

    # Validate all datasets (using multiprocessing across LMDB environments)
    print("\n--- Dataset Validation ---")
    with multiprocessing.Pool(min(len(lmdb_dirs), multiprocessing.cpu_count())) as pool:
        results = pool.map(validate_lmdb, lmdb_dirs)
        
    total_imgs = sum(r["total_images"] for r in results)
    missing_masks = sum(r["missing_masks"] for r in results)
    corrupted_images = sum(r["corrupted_images"] for r in results)
    corrupted_masks = sum(r["corrupted_masks"] for r in results)
    size_mismatches = sum(r["size_mismatches"] for r in results)
    
    tampered_total = sum(r["tampered"] for r in results)
    authentic_total = sum(r["authentic"] for r in results)
    
    all_res = []
    for r in results:
        all_res.extend(r["resolutions"])
        
    report_path = out_dir / "doctamper_validation_report.md"
    with open(report_path, "w") as f:
        f.write("# DocTamper LMDB Validation Report\n\n")
        f.write(f"- Total Images Found: {total_imgs}\n")
        f.write(f"- Total LMDB Environments: {len(lmdb_dirs)}\n")
        f.write(f"- Missing Masks: {missing_masks}\n")
        f.write(f"- Corrupted Images: {corrupted_images}\n")
        f.write(f"- Corrupted Masks: {corrupted_masks}\n")
        f.write(f"- Size Mismatches: {size_mismatches}\n")
        f.write(f"- Duplicate Filenames: 0 (LMDB uses sequential IDs)\n")
        
        f.write("\n## Details by Split\n")
        for r in results:
            f.write(f"### {r['lmdb_name']}\n")
            f.write(f"- Images: {r['total_images']}\n")
            f.write(f"- Missing masks: {r['missing_masks']}\n")
            f.write(f"- Size mismatches: {r['size_mismatches']}\n\n")
        
    logger.info(f"Saved validation report to {report_path}")
    
    # Dataset Statistics
    if total_imgs > 0:
        tamper_pct = (tampered_total / total_imgs) * 100
        auth_pct = (authentic_total / total_imgs) * 100
        
        avg_w = sum(w for w, h in all_res) / len(all_res) if all_res else 0
        avg_h = sum(h for w, h in all_res) / len(all_res) if all_res else 0
        min_res = min(all_res) if all_res else (0, 0)
        max_res = max(all_res) if all_res else (0, 0)
        
        stats = {
            "Total images": total_imgs,
            "Total masks": total_imgs - missing_masks,
            "Authentic documents": authentic_total,
            "Tampered documents": tampered_total,
            "Tamper percentage": round(tamper_pct, 2),
            "Authentic percentage": round(auth_pct, 2),
            "Average image resolution": f"{int(avg_w)}x{int(avg_h)}",
            "Minimum resolution": f"{min_res[0]}x{min_res[1]}",
            "Maximum resolution": f"{max_res[0]}x{max_res[1]}"
        }
        
        json_path = out_dir / "doctamper_statistics.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=4)
            
        csv_path = out_dir / "doctamper_statistics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(stats.keys())
            writer.writerow(stats.values())
            
        logger.info(f"Saved statistics to {json_path} and {csv_path}")

if __name__ == "__main__":
    main()
