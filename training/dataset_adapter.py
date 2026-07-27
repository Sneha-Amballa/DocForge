import sys
import io
from pathlib import Path
from PIL import Image
import numpy as np
from prompt_builder import build_qwen_prompt

sys.path.insert(0, str(Path("data").absolute()))
try:
    from verify_dataset import UnifiedDocForgeDataset
except ImportError:
    pass

class QwenDatasetAdapter:
    def __init__(self, split_file: str, doctamper_root: str, sroie_root: str):
        self.dataset = UnifiedDocForgeDataset(split_file, doctamper_root, sroie_root)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Convert image bytes or path to PIL Image
        if "image_bytes" in sample:
            image_obj = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")
        else:
            image_obj = Image.open(sample["image_path"]).convert("RGB")
            
        messages = build_qwen_prompt(sample, image_obj)
        return messages

def collate_qwen_batch(batch_messages, processor):
    """
    Takes a list of messages (from QwenDatasetAdapter) and processes them into model inputs.
    """
    from qwen_vl_utils import process_vision_info
    
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in batch_messages]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # We also need to extract labels for training. 
    # Qwen processor creates input_ids. For causal LM, labels are input_ids.
    # However, we should mask out the user prompt in the labels.
    # For now, to just perform a valid forward pass (which computes loss if labels are provided),
    # we'll clone input_ids to labels.
    inputs["labels"] = inputs["input_ids"].clone()
    
    # Mask padding tokens in labels
    if "attention_mask" in inputs:
        inputs["labels"][inputs["attention_mask"] == 0] = -100
        
    # Extract tampered flags for class-weighted loss
    import json
    tampered_flags = []
    for msg_list in batch_messages:
        assistant_content = msg_list[-1]["content"]
        try:
            data = json.loads(assistant_content)
            tampered_flags.append(data.get("tampered", False))
        except:
            tampered_flags.append(False)
            
    inputs["tampered_flags"] = tampered_flags
    return inputs
