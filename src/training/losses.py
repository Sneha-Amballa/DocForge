import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class LanguageModelingLoss(nn.Module):
    """Standard Causal Language Modeling Loss with ignored pad positions."""
    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Causal LM shifting
        # logits shape: (B, S, V) -> labels shape: (B, S)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )


class ClassificationLoss(nn.Module):
    """Binary Cross Entropy Loss for document tampering state classification."""
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: (B,) binary -> match logits: (B,)
        return self.loss_fn(logits.view(-1), targets.float().view(-1))


class LocalizationLoss(nn.Module):
    """L1 Loss on predicted bounding box coordinates."""
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred_coords, target_coords)


class DocForgeLossRegistry(nn.Module):
    """Registry aggregating multiple weighted losses (LM, Classification, and Localization)."""
    
    def __init__(
        self,
        lm_weight: float = 1.0,
        cls_weight: float = 0.0,
        loc_weight: float = 0.0
    ) -> None:
        """Initialize the loss manager.

        Args:
            lm_weight: Coefficient for Causal Language Modeling loss.
            cls_weight: Coefficient for Binary Classification loss.
            loc_weight: Coefficient for Bounding Box Localization loss.
        """
        super().__init__()
        self.lm_loss = LanguageModelingLoss()
        self.cls_loss = ClassificationLoss()
        self.loc_loss = LocalizationLoss()
        
        self.lm_weight = lm_weight
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight

    def forward(self, model_outputs: Any, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the weighted composite loss.

        Args:
            model_outputs: Raw output container from the VLM.
            batch: Data batch dictionary.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Aggregated Loss tensor, and dictionary of individual metrics.
        """
        loss_dict = {}
        device = batch["image_tensor"].device
        total_loss = torch.tensor(0.0, device=device)

        # 1. Language Modeling Loss
        if hasattr(model_outputs, "loss") and model_outputs.loss is not None:
            lm_val = model_outputs.loss
        else:
            # Fallback manual cross-entropy computation on logits
            logits = model_outputs.logits
            labels = batch["input_ids"] if "input_ids" in batch else torch.zeros(logits.shape[:2], dtype=torch.long, device=device)
            lm_val = self.lm_loss(logits, labels)

        total_loss = total_loss + self.lm_weight * lm_val
        loss_dict["lm_loss"] = lm_val.item()

        # 2. Classification Loss (tampering status)
        if self.cls_weight > 0.0 and "tampering_label" in batch:
            # Extract mock classification logits by pooling token predictions
            cls_logits = model_outputs.logits.mean(dim=1)[:, 0]
            cls_val = self.cls_loss(cls_logits, batch["tampering_label"])
            total_loss = total_loss + self.cls_weight * cls_val
            loss_dict["cls_loss"] = cls_val.item()

        # 3. Localization Loss (coordinate estimation)
        if self.loc_weight > 0.0 and "bbox" in batch:
            logits = model_outputs.logits
            # Project logits to box coordinates
            pred_box = logits.mean(dim=1)[:, :4]
            target_box = torch.zeros_like(pred_box)
            
            for idx, box_list in enumerate(batch["bbox"]):
                if len(box_list) > 0:
                    target_box[idx] = torch.tensor(box_list[0][:4], dtype=torch.float32, device=device)
                    
            loc_val = self.loc_loss(pred_box, target_box)
            total_loss = total_loss + self.loc_weight * loc_val
            loss_dict["loc_loss"] = loc_val.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
