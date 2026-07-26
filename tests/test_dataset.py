import unittest
from pathlib import Path
from PIL import Image

from src.config import DatasetConfig
from src.lmdb_reader import LMDBReader
from src.dataset import DocTamperDataset, MissingSampleError
from src.visualization import overlay_mask
from src.statistics import analyze_mask, format_statistics_table
from src.validation import verify_dataset

class TestDocTamperPipeline(unittest.TestCase):
    """Unit tests for DocForge Phase 0 Dataset Pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = DatasetConfig()
        cls.config.ensure_output_dirs()

    def test_config_paths(self) -> None:
        """Test if the configuration paths resolve correctly."""
        self.assertTrue(self.config.dataset_root.exists(), "Dataset root does not exist.")
        self.assertEqual(self.config.training_set.name, "DocTamperV1-TrainingSet")
        self.assertEqual(self.config.testing_set.name, "DocTamperV1-TestingSet")

    def test_lmdb_reader_basic(self) -> None:
        """Test if LMDBReader can open, read, and close safely."""
        db_path = self.config.training_set
        if not (db_path / "data.mdb").exists():
            self.skipTest(f"Training set not found at {db_path}")

        reader = LMDBReader(db_path)
        with reader as opened_reader:
            self.assertIsNotNone(opened_reader.env)
            # Try to read the num-samples metadata key
            num_samples = opened_reader.get_num_samples()
            self.assertGreater(num_samples, 0)
            
            # Read first image bytes
            img_bytes = opened_reader.get("image-000000000")
            self.assertIsNotNone(img_bytes)
            self.assertIsInstance(img_bytes, bytes)

    def test_dataset_indexing(self) -> None:
        """Test if DocTamperDataset supports PyTorch-like access."""
        db_path = self.config.training_set
        if not (db_path / "data.mdb").exists():
            self.skipTest(f"Training set not found at {db_path}")

        dataset = DocTamperDataset(db_path, config=self.config)
        with dataset:
            self.assertEqual(len(dataset), 120000)
            
            # Get sample 0
            sample = dataset[0]
            self.assertIn("image", sample)
            self.assertIn("mask", sample)
            self.assertIn("index", sample)
            self.assertEqual(sample["index"], 0)
            self.assertIsInstance(sample["image"], Image.Image)
            self.assertIsInstance(sample["mask"], Image.Image)
            
            # Check dimensions match
            self.assertEqual(sample["image"].size, sample["mask"].size)

    def test_dataset_out_of_bounds(self) -> None:
        """Test that indexing out of bounds raises IndexError."""
        db_path = self.config.training_set
        if not (db_path / "data.mdb").exists():
            self.skipTest(f"Training set not found at {db_path}")

        dataset = DocTamperDataset(db_path, config=self.config)
        with dataset:
            with self.assertRaises(IndexError):
                dataset.read_image(-1)
            with self.assertRaises(IndexError):
                dataset.read_image(9999999)

    def test_overlay_mask_blending(self) -> None:
        """Test that overlay blending creates correct sizes and modes."""
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        mask = Image.new("L", (100, 100), 0)
        # Put tampered pixels (value 255) in a section
        for x in range(30, 70):
            for y in range(30, 70):
                mask.putpixel((x, y), 255)

        overlay = overlay_mask(img, mask, alpha=0.5, color=(255, 0, 0))
        self.assertEqual(overlay.size, (100, 100))
        self.assertEqual(overlay.mode, "RGB")
        
        # Verify a pixel in the tampered zone has red blended in
        px = overlay.getpixel((50, 50))
        self.assertEqual(px[0], 255)  # R
        self.assertLess(px[1], 255)   # G (blended down)
        self.assertLess(px[2], 255)   # B (blended down)

    def test_statistics_helper(self) -> None:
        """Test single sample stats analyzer."""
        mask = Image.new("L", (100, 100), 0)
        # Put 2500 pixels as tampered (25% of 10000 pixels)
        for x in range(50):
            for y in range(50):
                mask.putpixel((x, y), 255)
                
        stats = analyze_mask(mask)
        self.assertEqual(stats["total_pixels"], 10000)
        self.assertEqual(stats["tampered_pixels"], 2500)
        self.assertEqual(stats["authentic_pixels"], 7500)
        self.assertAlmostEqual(stats["tampered_percentage"], 25.0)

    def test_validation_reports(self) -> None:
        """Test if the validation logic is run correctly."""
        db_path = self.config.training_set
        if not (db_path / "data.mdb").exists():
            self.skipTest(f"Training set not found at {db_path}")

        dataset = DocTamperDataset(db_path, config=self.config)
        report = verify_dataset(dataset, sample_limit=5)
        self.assertEqual(report["dataset_name"], "DocTamperV1-TrainingSet")
        self.assertEqual(report["checked_samples"], 5)
        self.assertTrue(report["passed"])

if __name__ == "__main__":
    unittest.main()
