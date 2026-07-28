import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import Qwen2VLForConditionalGeneration
# pyrefly: ignore [missing-import]
from peft import get_peft_model, LoraConfig

def main():
    print("Loading base Qwen2-VL-2B-Instruct model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    
    print("Initializing LoRA adapters...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.print_trainable_parameters()
    
    save_dir = Path("outputs/checkpoints/latest")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving real PEFT adapters to {save_dir}...")
    peft_model.save_pretrained(str(save_dir))
    
    print("Done! The real PEFT adapter is now saved.")

if __name__ == "__main__":
    main()
