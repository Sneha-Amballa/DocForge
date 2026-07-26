from pathlib import Path
from typing import Union, Tuple, List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from src.config import DatasetConfig
from src.dataset import DocTamperTorchDataset
from src.logger import get_logger

logger = get_logger("DocForge.DataLoader")

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for DocForge unified samples batching.

    Handles mixed tensor, PIL Image, string, and variable-length list values.

    Args:
        batch: List of dictionaries returned by DocTamperTorchDataset.

    Returns:
        Dict[str, Any]: Batched representations:
            - 'sample_id': List[str]
            - 'image_path': List[str]
            - 'image': List[PIL.Image]
            - 'mask': List[PIL.Image]
            - 'width': List[int]
            - 'height': List[int]
            - 'bbox': List[List[List[int]]] (variable size boxes per image)
            - 'normalized_bbox': List[List[List[int]]]
            - 'tampering_label': torch.LongTensor shape (B,)
            - 'forgery_type': List[str]
            - 'prompt': List[str]
            - 'image_tensor': torch.FloatTensor shape (B, 3, H, W)
            - 'mask_tensor': torch.FloatTensor shape (B, H, W)
            - 'metadata': List[Dict]
    """
    collate_dict: Dict[str, Any] = {
        "sample_id": [],
        "image_path": [],
        "image": [],
        "mask": [],
        "width": [],
        "height": [],
        "bbox": [],
        "normalized_bbox": [],
        "forgery_type": [],
        "prompt": [],
        "metadata": []
    }
    
    image_tensors = []
    mask_tensors = []
    labels = []
    
    for sample in batch:
        for k in collate_dict.keys():
            collate_dict[k].append(sample[k])
            
        image_tensors.append(sample["image_tensor"])
        mask_tensors.append(sample["mask_tensor"])
        labels.append(sample["tampering_label"])
        
    collate_dict["image_tensor"] = torch.stack(image_tensors, dim=0)
    collate_dict["mask_tensor"] = torch.stack(mask_tensors, dim=0)
    collate_dict["tampering_label"] = torch.tensor(labels, dtype=torch.long)
    
    return collate_dict

def get_dataloaders(
    db_path: Union[str, Path],
    config: Optional[DatasetConfig] = None,
    sample_limit: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split the dataset subset and create PyTorch DataLoaders for Train, Val, and Test.

    Ensures data augmentations are enabled only for training.

    Args:
        db_path: Path to the subset LMDB folder.
        config: Optional DatasetConfig loader.
        sample_limit: Optional limit to the number of samples to process (useful for dry runs).

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    cfg = config or DatasetConfig()
    
    # 1. Instantiate both datasets (one with augmentation, one without)
    train_dataset = DocTamperTorchDataset(db_path, config=cfg, augment=True)
    eval_dataset = DocTamperTorchDataset(db_path, config=cfg, augment=False)
    
    # Open database to fetch length
    train_dataset.open_database()
    total_samples = len(train_dataset)
    
    if sample_limit is not None:
        total_samples = min(sample_limit, total_samples)
        
    indices = np.arange(total_samples)
    
    # Seed splits for reproducibility
    np.random.seed(cfg.seed)
    np.random.shuffle(indices)
    
    train_end = int(round(cfg.train_ratio * total_samples))
    val_end = train_end + int(round(cfg.val_ratio * total_samples))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    logger.info(
        f"Splitting dataset of size {total_samples} into: "
        f"Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}"
    )
    
    # 2. Build Subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_dataset, val_indices)
    test_subset = Subset(eval_dataset, test_indices)
    
    # 3. Construct DataLoaders
    # Note: validation and testing subsets should not be shuffled
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader
