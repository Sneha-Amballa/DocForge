import unittest
from pathlib import Path
import tempfile
import shutil
import torch
import torch.nn as nn

from src.config import DatasetConfig
from src.model.config import VLMConfig, LoraConfigSettings
from src.model.qwen_model import load_base_vlm, freeze_base_parameters, MockQwen2VLForConditionalGeneration
from src.model.lora import attach_lora_adapters, get_trainable_parameters_summary
from src.model.utils import set_seed, save_lora_adapters, load_lora_adapters
from src.model.model_summary import print_model_parameter_summary

class TestVLMModule(unittest.TestCase):
    """Unit tests for Phase 2 Vision-Language Model and adapter layers module."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = DatasetConfig()
        cls.config.ensure_output_dirs()
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir)

    def test_vlm_config_init(self) -> None:
        """Test VLMConfig parameter loaders and device map setups."""
        vlm_cfg = VLMConfig(offline_mode=True, torch_dtype="float32")
        self.assertEqual(vlm_cfg.torch_dtype, torch.float32)
        self.assertTrue(vlm_cfg.offline_mode)
        self.assertIsNotNone(vlm_cfg.device)

        # Assert invalid dtype raises ValueError
        with self.assertRaises(ValueError):
            VLMConfig(torch_dtype="invalid_dtype")

    def test_lora_config_init(self) -> None:
        """Test LoRA config rank, alpha, targets, and bias setups."""
        lora_cfg = LoraConfigSettings(r=16, lora_alpha=32)
        self.assertEqual(lora_cfg.r, 16)
        self.assertEqual(lora_cfg.lora_alpha, 32)
        self.assertIn("q_proj", lora_cfg.target_modules)
        self.assertIn("gate_proj", lora_cfg.target_modules)

    def test_model_loading_and_freezing(self) -> None:
        """Test model load and parameter freezing checks."""
        vlm_cfg = VLMConfig(offline_mode=True)
        model, processor = load_base_vlm(vlm_cfg)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)
        self.assertIsInstance(model, MockQwen2VLForConditionalGeneration)

        # Count parameters before freezing
        summary_pre = get_trainable_parameters_summary(model)
        self.assertGreater(summary_pre["trainable_parameters"], 0)

        # Freeze parameters
        freeze_base_parameters(model)
        summary_post = get_trainable_parameters_summary(model)
        self.assertEqual(summary_post["trainable_parameters"], 0)
        self.assertEqual(summary_post["frozen_parameters"], summary_pre["total_parameters"])

    def test_lora_adapters_attachment(self) -> None:
        """Test that PEFT/Mock adapters attach and trainable parameter counts grow."""
        vlm_cfg = VLMConfig(offline_mode=True)
        model, _ = load_base_vlm(vlm_cfg)
        freeze_base_parameters(model)

        lora_cfg = LoraConfigSettings(r=8, lora_alpha=16)
        lora_model = attach_lora_adapters(model, lora_cfg)

        self.assertIsNotNone(lora_model)
        
        # Trainable parameters should now be greater than 0
        summary = get_trainable_parameters_summary(lora_model)
        self.assertGreater(summary["trainable_parameters"], 0)
        self.assertGreater(summary["frozen_parameters"], 0)
        self.assertGreater(summary["trainable_percentage"], 0.0)

    def test_forward_pass_and_logits(self) -> None:
        """Test mock forward pass and logits output shapes."""
        vlm_cfg = VLMConfig(offline_mode=True)
        model, _ = load_base_vlm(vlm_cfg)
        
        # Batch size 2, seq length 10
        input_ids = torch.zeros((2, 10), dtype=torch.long)
        
        # Forward pass on mock model
        outputs = model(input_ids)
        
        # Check shape of logits
        self.assertIsNotNone(outputs.logits)
        # Shape: (B, S, Vocab)
        self.assertEqual(outputs.logits.shape[:2], (2, 10))
        self.assertEqual(outputs.logits.shape[2], model.config.vocab_size)

    def test_adapter_save_and_load(self) -> None:
        """Test saving and re-loading adapter state weights."""
        vlm_cfg = VLMConfig(offline_mode=True)
        model, _ = load_base_vlm(vlm_cfg)
        freeze_base_parameters(model)

        lora_cfg = LoraConfigSettings(r=8, lora_alpha=16)
        lora_model = attach_lora_adapters(model, lora_cfg)
        
        # Save adapters
        save_path = Path(self.temp_dir) / "test_adapters"
        save_lora_adapters(lora_model, save_path)
        
        self.assertTrue((save_path / "adapter_model.bin").exists())
        self.assertTrue((save_path / "adapter_config.json").exists())

        # Load adapters back onto a fresh base model
        fresh_model, _ = load_base_vlm(vlm_cfg)
        freeze_base_parameters(fresh_model)
        
        loaded_model = load_lora_adapters(fresh_model, save_path)
        self.assertIsNotNone(loaded_model)
        
        summary = get_trainable_parameters_summary(loaded_model)
        self.assertGreater(summary["trainable_parameters"], 0)

if __name__ == "__main__":
    unittest.main()
