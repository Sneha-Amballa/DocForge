import os
import json
import csv
import torch
import logging
from pathlib import Path

class TrainingCallback:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "training_log.csv"
        self.json_log_file = self.output_dir / "training_log.json"
        
        self.logs = []
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "step", "loss", "learning_rate", "grad_norm", "gpu_memory_gb", "iteration_time"])

    def on_step_end(self, epoch, step, loss, lr, grad_norm, gpu_mem, iter_time):
        log_entry = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "learning_rate": lr,
            "grad_norm": grad_norm,
            "gpu_memory_gb": gpu_mem,
            "iteration_time": iter_time
        }
        self.logs.append(log_entry)
        
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, loss, lr, grad_norm, gpu_mem, iter_time])
            
        with open(self.json_log_file, "w") as f:
            json.dump(self.logs, f, indent=4)

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, checkpoint_dir, is_best=False):
    """
    Saves model LoRA adapters and optimizer states.
    """
    checkpoint_dir = Path(checkpoint_dir)
    target_dir = checkpoint_dir / "best" if is_best else checkpoint_dir / "last"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PEFT model
    model.save_pretrained(target_dir)
    
    # Save optimizer & scheduler
    torch.save(optimizer.state_dict(), target_dir / "optimizer.pt")
    if scheduler:
        torch.save(scheduler.state_dict(), target_dir / "scheduler.pt")
        
    # Save state
    state = {
        "epoch": epoch,
        "step": step,
        "loss": loss
    }
    with open(target_dir / "training_state.json", "w") as f:
        json.dump(state, f, indent=4)
        
    return target_dir
