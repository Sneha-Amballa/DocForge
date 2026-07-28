import io
from pathlib import Path
from typing import Union, Tuple, Dict, Any, List, Optional
# pyrefly: ignore [missing-import]
from PIL import Image

from src.config import DatasetConfig
from src.lmdb_reader import LMDBReader
from src.dataset import DocTamperDataset, DocTamperTorchDataset
from src.logger import get_logger
import src.visualization as vis
import src.statistics as stats
import src.validation as val
import src.dataset_loader as dl
import src.bbox_generator as bbox
import src.processor as proc

logger = get_logger("DocForge.Utils")

# Re-expose Phase 0 utilities
open_database = dl.DataLoader  # keeping naming helpers if needed, but we encourage OOP
def open_database_helper(db_path: Union[str, Path]) -> LMDBReader:
    reader = LMDBReader(db_path)
    return reader.open()

def close_database_helper(reader: LMDBReader) -> None:
    if reader is not None:
        reader.close()

# Expose key reader wrappers
read_image = DocTamperDataset.read_image
read_mask = DocTamperDataset.read_mask
read_sample = DocTamperDataset.read_sample

# Re-expose visualization functions
overlay_mask = vis.overlay_mask
show_image = vis.show_image
show_mask = vis.show_mask
show_overlay = vis.show_overlay
show_random_samples = vis.show_random_samples
visualize_preprocessed_sample = vis.visualize_preprocessed_sample

# Re-expose statistics and validation functions
dataset_statistics = stats.compute_dataset_statistics
verify_dataset = val.verify_dataset
validate_dataset_folders = val.validate_dataset_folders
run_prepipeline_validation = val.run_prepipeline_validation
generate_split_statistics_report = stats.generate_split_statistics_report

# Bounding box utilities
generate_bboxes_from_mask = bbox.generate_bboxes_from_mask
normalize_bbox_for_qwen2_vl = bbox.normalize_bbox_for_qwen2_vl
save_bboxes_to_json = bbox.save_bboxes_to_json

# Qwen2-VL Processor and Dataloader utilities
Qwen2VLDataProcessor = proc.Qwen2VLDataProcessor
collate_fn = dl.collate_fn
get_dataloaders = dl.get_dataloaders

def list_available_datasets(config: Optional[DatasetConfig] = None) -> List[str]:
    """List the available subsets that exist under the dataset root.

    Args:
        config: Optional DatasetConfig instance.

    Returns:
        List[str]: List of names of available datasets.
    """
    cfg = config or DatasetConfig()
    available = []
    
    subsets = {
        "TrainingSet": cfg.training_set,
        "TestingSet": cfg.testing_set,
        "SCD": cfg.scd_set,
        "FCD": cfg.fcd_set
    }
    
    for name, path in subsets.items():
        if path.exists():
            if (path / "data.mdb").exists():
                available.append(name)
                
    logger.info(f"Available datasets found: {available}")
    return available
