import io
from pathlib import Path
from typing import Union, Dict, Any, Tuple, Optional, List
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from src.lmdb_reader import LMDBReader
from src.logger import get_logger
from src.config import DatasetConfig
from src.bbox_generator import generate_bboxes_from_mask, normalize_bbox_for_qwen2_vl
from src.augmentations import DocForgeAugmentations

logger = get_logger("DocForge.Dataset")

class DatasetError(Exception):
    """Custom exception raised for dataset-related errors."""
    pass

class CorruptSampleError(DatasetError):
    """Exception raised when a sample's image or mask is corrupted."""
    pass

class MissingSampleError(DatasetError):
    """Exception raised when a requested sample is missing from the database."""
    pass


class DocTamperDataset:
    """A dataset class for DocTamper datasets stored in LMDB databases.

    Provides indexed access to images and masks. Supports lazy loading,
    validates database contents, and handles errors gracefully.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        config: Optional[DatasetConfig] = None
    ) -> None:
        """Initialize the DocTamper dataset.

        Args:
            db_path: Path to the LMDB database directory.
            config: Optional DatasetConfig object.
        """
        self.db_path = Path(db_path).resolve()
        self.config = config or DatasetConfig()
        self.reader: Optional[LMDBReader] = None
        self._length: Optional[int] = None

    def open_database(self) -> "DocTamperDataset":
        """Explicitly open the underlying database reader.

        Returns:
            DocTamperDataset: Self.
        """
        if self.reader is None:
            self.reader = LMDBReader(self.db_path)
            self.reader.open()
            logger.info(f"Dataset database opened for subset: {self.db_path.name}")
        return self

    def close_database(self) -> None:
        """Close the underlying database reader and release resources."""
        if self.reader is not None:
            self.reader.close()
            self.reader = None
            logger.info(f"Dataset database closed for subset: {self.db_path.name}")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        if self._length is None:
            if self.reader is None:
                self.open_database()
            assert self.reader is not None
            self._length = self.reader.get_num_samples()
        return self._length

    def read_image(self, index: int) -> Image.Image:
        """Retrieve and decode the original image at the given index.

        Args:
            index: 0-based sample index.

        Returns:
            Image.Image: Decoded Pillow image.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}.")

        if self.reader is None:
            self.open_database()
        
        assert self.reader is not None
        key = f"image-{index:09d}"
        
        img_bytes = self.reader.get(key)
        if img_bytes is None:
            raise MissingSampleError(f"Image key '{key}' not found for index {index}.")

        try:
            image = Image.open(io.BytesIO(img_bytes))
            image.load()
            return image
        except Exception as e:
            raise CorruptSampleError(f"Corrupt image at index {index} (key: {key}): {e}") from e

    def read_mask(self, index: int, normalize: bool = True) -> Image.Image:
        """Retrieve and decode the binary tampering mask at the given index.

        Args:
            index: 0-based sample index.
            normalize: If True, normalizes positive mask values (e.g. 1) to 255
                for downstream compatibility. Defaults to True.

        Returns:
            Image.Image: Decoded Pillow mask image (grayscale).
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}.")

        if self.reader is None:
            self.open_database()

        assert self.reader is not None
        key = f"label-{index:09d}"

        mask_bytes = self.reader.get(key)
        if mask_bytes is None:
            raise MissingSampleError(f"Mask key '{key}' not found for index {index}.")

        try:
            mask = Image.open(io.BytesIO(mask_bytes))
            mask.load()
            
            if normalize:
                mask_arr = np.array(mask.convert("L"))
                unique_vals = np.unique(mask_arr)
                if not np.all(np.isin(unique_vals, [0, 255])):
                    normalized_arr = (mask_arr > 0).astype(np.uint8) * 255
                    mask = Image.fromarray(normalized_arr)
                    
            return mask
        except Exception as e:
            raise CorruptSampleError(f"Corrupt mask at index {index} (key: {key}): {e}") from e

    def read_sample(self, index: int, normalize: bool = True) -> Tuple[Image.Image, Image.Image]:
        """Retrieve both image and mask for a sample.

        Args:
            index: 0-based sample index.
            normalize: Whether to normalize positive mask values to 255. Defaults to True.

        Returns:
            Tuple[Image.Image, Image.Image]: Original image and binary mask.
        """
        image = self.read_image(index)
        mask = self.read_mask(index, normalize=normalize)
        return image, mask

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Simple dict retrieval for backward compatibility."""
        try:
            image, mask = self.read_sample(index)
            return {
                "image": image,
                "mask": mask,
                "index": index
            }
        except DatasetError as e:
            raise KeyError(f"Failed to load sample at index {index}") from e

    def __enter__(self) -> "DocTamperDataset":
        self.open_database()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close_database()


class DocTamperTorchDataset(Dataset):
    """PyTorch Dataset subclass wrapper around DocTamperDataset.

    Standardizes inputs for downstream model pipelines (such as Qwen2-VL),
    supports configurable data augmentations, aspect ratio preservation resizing,
    optional padding, and formats bounding boxes extracted from masks.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        config: Optional[DatasetConfig] = None,
        augment: bool = False
    ) -> None:
        """Initialize the PyTorch wrapper.

        Args:
            db_path: Path to LMDB database subset.
            config: DatasetConfig instance.
            augment: Force enable/disable data augmentations.
        """
        self.config = config or DatasetConfig()
        self.raw_dataset = DocTamperDataset(self.config.get_subset_path(str(db_path)) if not Path(db_path).exists() else db_path, config=self.config)
        self.augment = augment and self.config.aug_enabled
        self.augmenter = DocForgeAugmentations(self.config)

    def open_database(self) -> "DocTamperTorchDataset":
        """Open LMDB database reader."""
        self.raw_dataset.open_database()
        return self

    def close_database(self) -> None:
        """Close LMDB database reader."""
        self.raw_dataset.close_database()

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def _resize_and_pad(
        self,
        img: Image.Image,
        target_size: Tuple[int, int],
        is_mask: bool = False
    ) -> Image.Image:
        """Resize and pad the image or mask preserving aspect ratio.

        Args:
            img: Pillow image to process.
            target_size: Width and height of target dimensions.
            is_mask: Whether this is a binary mask (uses Nearest Neighbor).

        Returns:
            Image.Image: Resized (and padded) PIL Image.
        """
        target_w, target_h = target_size
        
        if not self.config.preserve_aspect_ratio:
            resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
            return img.resize(target_size, resample)
            
        w, h = img.size
        # Calculate aspect scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        
        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
        resized = img.resize((new_w, new_h), resample)
        
        if not self.config.padding_enabled:
            return resized
            
        # Draw on padded canvas
        if is_mask:
            canvas = Image.new("L", target_size, 0)
        else:
            canvas = Image.new("RGB", target_size, (128, 128, 128))
            
        # Paste centered or top-left (we stick to top-left (0,0) for unified coordinate consistency)
        canvas.paste(resized, (0, 0))
        return canvas

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load and process sample at index.

        Args:
            index: Sample index integer.

        Returns:
            Dict[str, Any]: Unified sample representation containing:
                - 'sample_id': Unique identifier string
                - 'image_path': Descriptor path
                - 'image': Preprocessed PIL Image (RGB)
                - 'width': Original image width
                - 'height': Original image height
                - 'mask': Preprocessed PIL mask Image (binary)
                - 'bbox': Absolute bounding boxes list [[xmin, ymin, xmax, ymax], ...]
                - 'normalized_bbox': Bounding boxes normalized to [ymin, xmin, ymax, xmax] (0-1000)
                - 'tampering_label': 1 if tampered, 0 if authentic
                - 'forgery_type': String category of forgery
                - 'prompt': Instruction text
                - 'image_tensor': FloatTensor of shape (3, H, W)
                - 'mask_tensor': FloatTensor of shape (H, W)
                - 'metadata': Context dictionaries
        """
        # Load raw PIL images
        raw_img = self.raw_dataset.read_image(index).convert("RGB")
        raw_msk = self.raw_dataset.read_mask(index, normalize=True).convert("L")
        
        orig_w, orig_h = raw_img.size
        
        # Determine tampering state from raw mask
        mask_arr = np.array(raw_msk)
        has_tampering = bool(np.any(mask_arr > 0))
        tampering_label = 1 if has_tampering else 0
        
        # Determine forgery type based on subset
        db_name = self.raw_dataset.db_path.name.lower()
        if "scd" in db_name:
            forgery_type = "Single-page Copy-Move"
        elif "fcd" in db_name:
            forgery_type = "Full-page Copy-Move"
        else:
            forgery_type = "Copy-Move" if tampering_label == 1 else "Authentic"
            
        # 1. Bounding Box Generation (drawn from raw mask)
        absolute_bboxes = generate_bboxes_from_mask(raw_msk)
        normalized_bboxes = [
            normalize_bbox_for_qwen2_vl(box, orig_w, orig_h)
            for box in absolute_bboxes
        ]
        
        # 2. Apply Spatial Data Augmentation (if enabled)
        aug_img, aug_msk = raw_img, raw_msk
        if self.augment:
            aug_img, aug_msk = self.augmenter.apply(raw_img, raw_msk)
            
        # 3. Resize and Padding Preprocessing
        proc_img = self._resize_and_pad(aug_img, self.config.image_size, is_mask=False)
        proc_msk = self._resize_and_pad(aug_msk, self.config.image_size, is_mask=True)
        
        # 4. Conversion to Tensors
        img_np = np.array(proc_img).astype(np.float32) / 255.0
        # Normalize
        mean = np.array(self.config.normalization_mean, dtype=np.float32)
        std = np.array(self.config.normalization_std, dtype=np.float32)
        img_np = (img_np - mean) / std
        # HWC to CHW
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
        
        # Mask tensor
        msk_np = (np.array(proc_msk) > 127).astype(np.float32)
        msk_tensor = torch.from_numpy(msk_np)
        
        sample_id = f"{self.raw_dataset.db_path.name}_{index:09d}"
        image_path = f"LMDB://{self.raw_dataset.db_path.name}/image-{index:09d}"
        
        return {
            "sample_id": sample_id,
            "image_path": image_path,
            "image": proc_img,
            "width": orig_w,
            "height": orig_h,
            "mask": proc_msk,
            "bbox": absolute_bboxes,
            "normalized_bbox": normalized_bboxes,
            "tampering_label": tampering_label,
            "forgery_type": forgery_type,
            "prompt": self.config.prompt_template,
            "image_tensor": img_tensor,
            "mask_tensor": msk_tensor,
            "metadata": {
                "index": index,
                "subset": self.raw_dataset.db_path.name,
                "original_size": (orig_w, orig_h)
            }
        }

    def __enter__(self) -> "DocTamperTorchDataset":
        self.open_database()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close_database()
