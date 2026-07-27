import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

def compute_weighted_loss(logits, labels, tampered_flags, class_weights):
    """
    Computes per-sample Causal LM loss and applies class weights.
    class_weights: dict e.g., {True: 0.1, False: 10.0}
    tampered_flags: list of bools for the batch indicating if the sample is tampered.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size(0), -1)
    
    # Mean loss per sample, ignoring -100
    valid_tokens = (shift_labels != -100).float()
    sample_losses = (loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1e-5)
    
    device = logits.device
    weights = torch.tensor([class_weights[flag] for flag in tampered_flags], device=device)
    
    weighted_loss = (sample_losses * weights).mean()
    return weighted_loss

def compute_focal_loss(logits, labels, tampered_flags, class_weights, gamma=2.0):
    """
    Computes Focal Loss for Causal LM.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = CrossEntropyLoss(reduction='none')
    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).view(shift_labels.size(0), -1)
    
    valid_tokens = (shift_labels != -100).float()
    sample_losses = (focal_loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1e-5)
    
    device = logits.device
    weights = torch.tensor([class_weights[flag] for flag in tampered_flags], device=device)
    
    weighted_focal_loss = (sample_losses * weights).mean()
    return weighted_focal_loss
