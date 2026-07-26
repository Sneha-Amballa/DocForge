import unittest
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.config import DatasetConfig
from src.bbox_generator import generate_bboxes_from_mask, normalize_bbox_for_qwen2_vl
from src.augmentations import DocForgeAugmentations
from src.dataset import DocTamperTorchDataset
from src.processor import Qwen2VLDataProcessor
from src.dataset_loader import get_dataloaders, collate_fn
from src.validation import validate_dataset_folders, run_prepipeline_validation

class TestPreprocessingPipeline(unittest.TestCase):
    """Unit tests for Phase 1 Preprocessing and Data Preparation Pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = DatasetConfig()
        cls.config.ensure_output_dirs()

    def test_yaml_config_loading(self) -> None:
        """Test that configuration parameters load correctly from YAML."""
        self.assertIsNotNone(self.config.image_size)
        self.assertIsInstance(self.config.image_size, tuple)
        self.assertEqual(len(self.config.image_size), 2)
        self.assertIsInstance(self.config.preserve_aspect_ratio, bool)
        self.assertIsInstance(self.config.padding_enabled, bool)
        self.assertIsInstance(self.config.aug_enabled, bool)
        self.assertGreater(self.config.batch_size, 0)

    def test_bbox_generation_single_region(self) -> None:
        """Test bbox extraction from a mask with a single tampered block."""
        # Create a blank mask (100x100)
        mask = Image.new("L", (100, 100), 0)
        # Add a tampered block at x: 20-50, y: 30-70
        for x in range(20, 50):
            for y in range(30, 70):
                mask.putpixel((x, y), 255)

        bboxes = generate_bboxes_from_mask(mask)
        self.assertEqual(len(bboxes), 1)
        # Expected bbox: [xmin, ymin, xmax, ymax]
        # xmax, ymax are exclusive (indices range is 20-49 and 30-69, so boundary is [20, 30, 50, 70])
        self.assertEqual(bboxes[0], [20, 30, 50, 70])

    def test_bbox_generation_multiple_regions(self) -> None:
        """Test bbox extraction from a mask with multiple disjoint regions."""
        mask = Image.new("L", (100, 100), 0)
        
        # Region 1: [10, 10, 30, 30]
        for x in range(10, 30):
            for y in range(10, 30):
                mask.putpixel((x, y), 255)
                
        # Region 2: [60, 60, 80, 90]
        for x in range(60, 80):
            for y in range(60, 90):
                mask.putpixel((x, y), 255)

        bboxes = generate_bboxes_from_mask(mask)
        # Expect exactly 2 boxes
        self.assertEqual(len(bboxes), 2)
        sorted_bboxes = sorted(bboxes, key=lambda b: b[0])
        self.assertEqual(sorted_bboxes[0], [10, 10, 30, 30])
        self.assertEqual(sorted_bboxes[1], [60, 60, 80, 90])

    def test_bbox_normalization(self) -> None:
        """Test scaling of absolute coordinates to Qwen2-VL's normalized 1000 scale."""
        bbox = [100, 200, 300, 400]
        # Image size 400x800 (width, height)
        # scale factors: width=400, height=800
        # Qwen format: [ymin, xmin, ymax, xmax] scaled to 1000
        # ymin_norm = 200 / 800 * 1000 = 250
        # xmin_norm = 100 / 400 * 1000 = 250
        # ymax_norm = 400 / 800 * 1000 = 500
        # xmax_norm = 300 / 400 * 1000 = 750
        norm_bbox = normalize_bbox_for_qwen2_vl(bbox, img_width=400, img_height=800)
        self.assertEqual(norm_bbox, [250, 250, 500, 750])

    def test_augmentations_spatial_alignment(self) -> None:
        """Test that augmentations transform image and mask and keep sizes matched."""
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        mask = Image.new("L", (200, 200), 0)
        for x in range(50, 150):
            for y in range(50, 150):
                mask.putpixel((x, y), 255)

        # Force enable augmentations setting
        self.config.aug_enabled = True
        augmenter = DocForgeAugmentations(self.config)
        
        aug_img, aug_mask = augmenter.apply(img, mask)
        self.assertIsNotNone(aug_mask)
        # Check size alignment
        self.assertEqual(aug_img.size, aug_mask.size)

    def test_processor_fallback(self) -> None:
        """Test that the Qwen2-VL processor class initiates and processes in fallback mode."""
        processor = Qwen2VLDataProcessor(processor_name="dummy_invalid_processor_path")
        self.assertTrue(processor.is_fallback)
        self.assertIsNone(processor.processor)

        # Create dummy preprocessed unified sample dictionary
        sample = {
            "sample_id": "dummy_0",
            "image_path": "dummy://path",
            "image": Image.new("RGB", (512, 512)),
            "width": 512,
            "height": 512,
            "mask": Image.new("L", (512, 512)),
            "bbox": [[100, 100, 200, 200]],
            "normalized_bbox": [[200, 200, 400, 400]],
            "tampering_label": 1,
            "forgery_type": "Copy-Move",
            "prompt": "Detect document tampering.",
            "image_tensor": torch.zeros((3, 512, 512), dtype=torch.float32),
            "mask_tensor": torch.zeros((512, 512), dtype=torch.float32)
        }

        inputs = processor.process_sample(sample)
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_grid_thw", inputs)
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 512, 512))

    def test_dataloaders_creation(self) -> None:
        """Test splitting datasets and compiling batches in DataLoader."""
        db_path = self.config.training_set
        if not (db_path / "data.mdb").exists():
            self.skipTest(f"Training set database not found at {db_path}")

        train_loader, val_loader, test_loader = get_dataloaders(
            db_path,
            config=self.config,
            sample_limit=20
        )

        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

        # Retrieve a batch
        batch = next(iter(train_loader))
        self.assertIn("image_tensor", batch)
        self.assertIn("mask_tensor", batch)
        self.assertIn("tampering_label", batch)
        
        # Check shapes
        B = self.config.batch_size
        self.assertEqual(batch["image_tensor"].shape, (B, 3, 512, 512))
        self.assertEqual(batch["mask_tensor"].shape, (B, 512, 512))
        self.assertEqual(batch["tampering_label"].shape, (B,))

    def test_prepipeline_validation(self) -> None:
        """Test prepipeline directory validation execution."""
        status = validate_dataset_folders(self.config)
        self.assertIn("TrainingSet", status)
        self.assertIn("TestingSet", status)
        self.assertIn("SCD", status)
        self.assertIn("FCD", status)

if __name__ == "__main__":
    unittest.main()
