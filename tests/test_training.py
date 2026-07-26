import unittest
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn

from src.config import DatasetConfig
from src.model.qwen_model import MockQwen2VLForConditionalGeneration, freeze_base_parameters
from src.model.lora import attach_lora_adapters, get_trainable_parameters_summary
from src.model.config import LoraConfigSettings
from src.training.losses import DocForgeLossRegistry, LanguageModelingLoss, ClassificationLoss
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.training.checkpoint import save_checkpoint, load_checkpoint

class TestTrainingPipeline(unittest.TestCase):
    """Unit tests for Phase 3 Training and Fine-Tuning Pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = DatasetConfig()
        cls.config.ensure_output_dirs()
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir)

    def test_loss_registry_computations(self) -> None:
        """Test modular loss registry components compute correct metrics."""
        loss_registry = DocForgeLossRegistry(lm_weight=1.0, cls_weight=1.0, loc_weight=1.0)
        
        # Batch size 2, sequence length 5, vocab 100
        logits = torch.randn(2, 5, 100)
        input_ids = torch.randint(0, 100, (2, 5))
        tampering_label = torch.tensor([1, 0])
        bbox = [[[10, 20, 30, 40]], [[50, 60, 70, 80]]]
        image_tensor = torch.zeros(2, 3, 32, 32)
        
        class DummyOutputs:
            def __init__(self, logits: torch.Tensor) -> None:
                self.logits = logits
                self.loss = None
                
        outputs = DummyOutputs(logits)
        batch = {
            "input_ids": input_ids,
            "tampering_label": tampering_label,
            "bbox": bbox,
            "image_tensor": image_tensor
        }
        
        total_loss, loss_dict = loss_registry(outputs, batch)
        
        self.assertIsNotNone(total_loss)
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn("lm_loss", loss_dict)
        self.assertIn("cls_loss", loss_dict)
        self.assertIn("loc_loss", loss_dict)
        self.assertIn("total_loss", loss_dict)

    def test_optimizer_trainable_parameters(self) -> None:
        """Test optimizer setup filters out frozen weights and optimizes LoRA weights."""
        model = MockQwen2VLForConditionalGeneration()
        freeze_base_parameters(model)
        
        # Expect ValueError if no LoRA adapters are attached
        with self.assertRaises(ValueError):
            get_optimizer(model)
            
        # Attach adapters
        lora_cfg = LoraConfigSettings()
        model_lora = attach_lora_adapters(model, lora_cfg)
        
        optimizer = get_optimizer(model_lora, learning_rate=0.0001, weight_decay=0.01)
        self.assertIsNotNone(optimizer)
        self.assertEqual(len(optimizer.param_groups), 2)
        
        # Check that base parameters are not in the optimizer parameter groups
        for group in optimizer.param_groups:
            for param in group["params"]:
                self.assertTrue(param.requires_grad)

    def test_scheduler_warmup_and_decay(self) -> None:
        """Test scheduler learning rate scaling math."""
        model = MockQwen2VLForConditionalGeneration()
        freeze_base_parameters(model)
        lora_cfg = LoraConfigSettings()
        model_lora = attach_lora_adapters(model, lora_cfg)
        
        optimizer = get_optimizer(model_lora)
        scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_type="linear",
            num_warmup_steps=10,
            num_training_steps=100
        )
        
        # Warmup phase (scaling up)
        optimizer.step()
        lrs = []
        for i in range(15):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            
        # First 10 steps should increase LR
        self.assertLess(lrs[0], lrs[9])
        # After step 10, LR should decrease
        self.assertGreater(lrs[9], lrs[14])

    def test_checkpoint_saving_and_loading(self) -> None:
        """Test checkpointer properly dumps state dictionaries and seeds."""
        model = MockQwen2VLForConditionalGeneration()
        freeze_base_parameters(model)
        lora_cfg = LoraConfigSettings()
        model_lora = attach_lora_adapters(model, lora_cfg)
        
        optimizer = get_optimizer(model_lora)
        scheduler = get_scheduler(optimizer, num_warmup_steps=5, num_training_steps=50)
        
        save_path = Path(self.temp_dir) / "test_checkpointer"
        
        metrics = {"loss": 0.35, "train_loss": 0.42}
        checkpoint_dir = save_checkpoint(
            model=model_lora,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            metrics=metrics,
            checkpoints_dir=save_path,
            is_best=True
        )
        
        self.assertTrue((checkpoint_dir / "adapter_model.bin").exists())
        self.assertTrue((checkpoint_dir / "training_state.pth").exists())
        self.assertTrue((save_path / "latest" / "training_state.pth").exists())
        self.assertTrue((save_path / "best" / "training_state.pth").exists())

        # Load back checkpoint onto fresh instances
        fresh_model = MockQwen2VLForConditionalGeneration()
        freeze_base_parameters(fresh_model)
        fresh_optimizer = get_optimizer(attach_lora_adapters(fresh_model, lora_cfg))
        fresh_scheduler = get_scheduler(fresh_optimizer, num_warmup_steps=5, num_training_steps=50)
        
        meta = load_checkpoint(
            model=fresh_model,
            optimizer=fresh_optimizer,
            scheduler=fresh_scheduler,
            checkpoint_path=save_path / "latest"
        )
        
        self.assertEqual(meta["epoch"], 2)
        self.assertEqual(meta["metrics"]["loss"], 0.35)

if __name__ == "__main__":
    unittest.main()
