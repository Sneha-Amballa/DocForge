import time
import torch
import json
from tqdm import tqdm
from callbacks import TrainingCallback, save_checkpoint
from losses import compute_weighted_loss, compute_focal_loss

class QwenTrainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config, class_weights, logger):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.class_weights = class_weights
        self.logger = logger
        
        self.device = next(model.parameters()).device
        self.callback = TrainingCallback(config.get("output_dir", "./outputs/training"))
        
        self.use_class_weighted_loss = config.get("class_weighted_loss", True)
        self.use_focal_loss = config.get("focal_loss", False)
        
    def train(self):
        epochs = self.config.get("epochs", 1)
        grad_accum = self.config.get("gradient_accumulation_steps", 1)
        
        self.logger.info("✓ training started")
        
        global_step = 0
        best_val_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")
            for step, batch in enumerate(pbar):
                start_time = time.time()
                
                # Move to device
                tampered_flags = batch.pop("tampered_flags", None)
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
                
                outputs = self.model(**inputs, output_hidden_states=False)
                
                if self.use_focal_loss and tampered_flags is not None:
                    loss = compute_focal_loss(outputs.logits, inputs["labels"], tampered_flags, self.class_weights)
                elif self.use_class_weighted_loss and tampered_flags is not None:
                    loss = compute_weighted_loss(outputs.logits, inputs["labels"], tampered_flags, self.class_weights)
                else:
                    loss = outputs.loss
                    
                loss = loss / grad_accum
                loss.backward()
                
                if (step + 1) % grad_accum == 0 or (step + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                iter_time = time.time() - start_time
                gpu_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                lr = self.scheduler.get_last_lr()[0]
                
                pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}", "lr": f"{lr:.2e}"})
                epoch_loss += loss.item() * grad_accum
                
                self.callback.on_step_end(epoch, global_step, loss.item() * grad_accum, lr, 0.0, gpu_mem, iter_time)
                
            val_loss = self.validate()
            self.logger.info(f"Epoch {epoch} | Train Loss: {epoch_loss/len(self.train_loader):.4f} | Val Loss: {val_loss:.4f}")
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                
            save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, global_step, val_loss, self.config.get("checkpoint_dir", "./outputs/checkpoints"), is_best)
            self.logger.info("✓ checkpoint saved")
            
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch.pop("tampered_flags", None)
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
                outputs = self.model(**inputs)
                total_loss += outputs.loss.item()
        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
