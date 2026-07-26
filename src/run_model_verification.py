import sys
from pathlib import Path
import torch

# Ensure project root is in system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DatasetConfig
from src.model.config import VLMConfig, LoraConfigSettings
from src.model.qwen_model import load_base_vlm, freeze_base_parameters
from src.model.lora import attach_lora_adapters, get_trainable_parameters_summary
from src.model.model_summary import print_model_parameter_summary
from src.model.utils import set_seed
from src.dataset import DocTamperTorchDataset
from src.logger import get_logger

logger = get_logger("DocForge.Verification")

def run_verification(offline: bool = True) -> None:
    """Orchestrates Phase 2 validations and prints a pass/fail summary table."""
    print("=" * 70)
    print("STARTING VISION-LANGUAGE MODEL PIPELINE VERIFICATION")
    print("=" * 70)

    # Set seed for reproducibility
    set_seed(42)

    checks = {
        "1. Base Model Loading": False,
        "2. AutoProcessor Loading": False,
        "3. Dataset Sample Processing": False,
        "4. LoRA Adapter Attachment": False,
        "5. Parameter Freezing": False,
        "6. Trainable Parameters Check": False,
        "7. Model Forward Pass": False,
        "8. Inference Generation": False
    }

    # Configuration loaders
    config = DatasetConfig()
    vlm_config = VLMConfig(offline_mode=offline, dataset_config=config)
    lora_config = LoraConfigSettings()

    model = None
    processor = None
    sample = None
    lora_model = None

    # Step 1 & 2: Load Model and Processor
    try:
        model, processor = load_base_vlm(vlm_config)
        checks["1. Base Model Loading"] = model is not None
        checks["2. AutoProcessor Loading"] = processor is not None
    except Exception as e:
        print(f"ERROR: Model/Processor loading failed: {e}")

    # Step 3: Dataset Sample Processing
    try:
        # Load the dataset
        dataset = DocTamperTorchDataset(config.training_set, config=config, augment=False)
        with dataset:
            # Retrieve sample 0
            sample = dataset[0]
            
        print("\nUnified Sample Representation keys:")
        print(list(sample.keys()))
        print(f"Prompt text: \"{sample['prompt']}\"")
        print(f"Original shape: {sample['width']}x{sample['height']}")
        print(f"Processed image size: {sample['image'].size}")
        
        checks["3. Dataset Sample Processing"] = sample is not None
    except Exception as e:
        print(f"ERROR: Dataset sample processing failed: {e}")

    # Step 5: Freeze Base Parameters
    if model is not None:
        try:
            # Count parameters before LoRA
            pre_summary = get_trainable_parameters_summary(model)
            print(f"\nBefore Freezing - Trainable: {pre_summary['trainable_parameters']:,} || Total: {pre_summary['total_parameters']:,}")
            
            freeze_base_parameters(model)
            
            post_freeze_summary = get_trainable_parameters_summary(model)
            print(f"After Freezing - Trainable: {post_freeze_summary['trainable_parameters']:,} || Total: {post_freeze_summary['total_parameters']:,}")
            
            checks["5. Parameter Freezing"] = post_freeze_summary["trainable_parameters"] == 0
        except Exception as e:
            print(f"ERROR: Freezing parameters failed: {e}")

    # Step 4 & 6: Attach LoRA adapters and check trainable parameters
    if model is not None:
        try:
            lora_model = attach_lora_adapters(model, lora_config)
            checks["4. LoRA Adapter Attachment"] = lora_model is not None
            
            # Count trainable parameters
            post_lora_summary = get_trainable_parameters_summary(lora_model)
            print("\nTrainable parameters summary post-LoRA:")
            print_model_parameter_summary(lora_model)
            
            # Checks if trainable parameters count is greater than zero and frozen count is greater than zero
            checks["6. Trainable Parameters Check"] = (
                post_lora_summary["trainable_parameters"] > 0 and 
                post_lora_summary["frozen_parameters"] > 0
            )
        except Exception as e:
            print(f"ERROR: LoRA adapter attachment failed: {e}")

    # Step 7: Model Forward Pass
    if lora_model is not None and processor is not None and sample is not None:
        try:
            print("\nRunning VLM batch processor...")
            # Prepare inputs using processor
            vlm_inputs = processor.process_sample(sample)
            
            # Transfer input tensors to device
            device = vlm_config.device
            input_ids = vlm_inputs["input_ids"].to(device)
            attention_mask = vlm_inputs["attention_mask"].to(device)
            pixel_values = vlm_inputs["pixel_values"].to(device)
            image_grid_thw = vlm_inputs["image_grid_thw"].to(device)
            
            print("Tensor shapes ready for forward pass:")
            print(f"  input_ids: {list(input_ids.shape)}")
            print(f"  attention_mask: {list(attention_mask.shape)}")
            print(f"  pixel_values: {list(pixel_values.shape)}")
            print(f"  image_grid_thw: {list(image_grid_thw.shape)}")
            
            # Execute forward pass without gradient calculations
            lora_model.eval()
            with torch.no_grad():
                outputs = lora_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw
                )
            
            logits = outputs.logits
            print(f"Forward pass completed successfully. Logits shape: {list(logits.shape)}")
            checks["7. Model Forward Pass"] = logits is not None and len(logits.shape) == 3
        except Exception as e:
            print(f"ERROR: Forward pass execution failed: {e}")

    # Step 8: Inference Generation
    if lora_model is not None and processor is not None and sample is not None:
        try:
            print("\nRunning sample inference generation...")
            vlm_inputs = processor.process_sample(sample)
            input_ids = vlm_inputs["input_ids"].to(vlm_config.device)
            
            lora_model.eval()
            with torch.no_grad():
                # Call generate method
                output_ids = lora_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=10
                )
            
            print(f"Generated output tokens shape: {list(output_ids.shape)}")
            print(f"Output tokens list: {output_ids[0].tolist()}")
            checks["8. Inference Generation"] = output_ids is not None and output_ids.shape[0] > 0
        except Exception as e:
            print(f"ERROR: Inference generation failed: {e}")

    # Display pass/fail summary matrix
    print("\n" + "=" * 50)
    print(f"| {'VERIFICATION CHECK LIST':^46} |")
    print("=" * 50)
    for name, passed in checks.items():
        status = "PASSED" if passed else "FAILED"
        print(f"| {name:<35} | {status:^8} |")
    print("=" * 50)

    # Determine exit code
    all_passed = all(checks.values())
    if not all_passed:
        logger.error("Some model verification checks failed.")
        sys.exit(1)
    else:
        logger.info("All pipeline verification checks passed.")

if __name__ == "__main__":
    # Force offline mock mode for checks by default
    offline_flag = True
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ["false", "online"]:
            offline_flag = False
            
    run_verification(offline=offline_flag)
