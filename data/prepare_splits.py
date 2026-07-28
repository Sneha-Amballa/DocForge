import os
import sys
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

# Add eval and data to sys path to import the existing loaders
sys.path.insert(0, str(Path("eval").absolute()))
sys.path.insert(0, str(Path("data").absolute()))

try:
    from load_doctamper import LMDBDataset
    from dataset_utils import get_lmdb_envs, setup_logger
    from load_sroie2019 import SROIEDataset
except ImportError as e:
    print(f"Error importing dataset loaders: {e}")
    sys.exit(1)

from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doctamper_root", type=str, default="./data")
    parser.add_argument("--sroie_root", type=str, default="C:\\Users\\USER\\.cache\\kagglehub\\datasets\\urbikn\\sroie-datasetv2\\versions\\4")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("prepare_splits")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(args.seed)
    
    logger.info("Discovering DocTamper LMDB environments...")
    lmdb_dirs = get_lmdb_envs(args.doctamper_root)
    doctamper_samples = []
    
    for lmdb_dir in lmdb_dirs:
        # Fast metadata collection without reading images
        logger.info(f"Indexing {lmdb_dir.name}...")
        dataset = LMDBDataset(str(lmdb_dir))
        length = len(dataset)
        for idx in range(length):
            doctamper_samples.append({
                "source": "doctamper",
                "lmdb_path": str(lmdb_dir.absolute()),
                "index": idx,
                "tampered": True,
                "forgery_type": "unknown"
            })
            
    logger.info("Discovering SROIE2019 dataset...")
    sroie_dataset = SROIEDataset(args.sroie_root)
    sroie_samples = []
    for idx in range(len(sroie_dataset)):
        sroie_samples.append({
            "source": "sroie2019",
            "index": idx,
            "tampered": False,
            "forgery_type": None
        })
        
    all_samples = doctamper_samples + sroie_samples
    logger.info(f"Total samples indexed: {len(all_samples)}")
    
    # Extract labels for stratification
    labels = [s["tampered"] for s in all_samples]
    
    logger.info(f"Performing stratified split (val_ratio={args.val_ratio}, seed={args.seed})...")
    train_samples, val_samples = train_test_split(
        all_samples, 
        test_size=args.val_ratio, 
        stratify=labels, 
        random_state=args.seed
    )
    
    train_split_path = out_dir / "train_split.json"
    val_split_path = out_dir / "val_split.json"
    
    with open(train_split_path, "w") as f:
        json.dump(train_samples, f)
    with open(val_split_path, "w") as f:
        json.dump(val_samples, f)
        
    logger.info(f"Saved train split to {train_split_path}")
    logger.info(f"Saved val split to {val_split_path}")
    
    # Save config
    config = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "doctamper_root": args.doctamper_root,
        "sroie_root": args.sroie_root,
        "shuffle": True
    }
    with open(out_dir / "split_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # Generate statistics
    def compute_stats(samples, name):
        tampered = sum(1 for s in samples if s["tampered"])
        authentic = len(samples) - tampered
        
        stats = {
            "Total samples": len(samples),
            "Tampered count": tampered,
            "Authentic count": authentic,
            "Tampered percentage": round((tampered / len(samples)) * 100, 2) if samples else 0,
            "Authentic percentage": round((authentic / len(samples)) * 100, 2) if samples else 0
        }
        
        stats_path = out_dir / f"{name}_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Saved {name} statistics to {stats_path}")
        return stats
        
    train_stats = compute_stats(train_samples, "train")
    val_stats = compute_stats(val_samples, "validation")
    
    print("\n================ SPLIT STATISTICS ================")
    print("TRAIN SET:")
    for k, v in train_stats.items(): print(f"  {k}: {v}")
    print("\nVALIDATION SET:")
    for k, v in val_stats.items(): print(f"  {k}: {v}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
