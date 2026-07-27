from peft import LoraConfig, get_peft_model, TaskType
import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
    return logger

def attach_lora_to_model(model, config: dict):
    """
    Attaches LoRA adapters to the base model using the given configuration.
    """
    logger = setup_logger("lora_utils")
    
    lora_config = LoraConfig(
        r=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    logger.info("Attaching LoRA adapters to model...")
    peft_model = get_peft_model(model, lora_config)
    
    logger.info("✓ LoRA attached successfully.")
    
    trainable_params = 0
    all_param = 0
    for _, param in peft_model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    frozen_params = all_param - trainable_params
    pct_trainable = 100 * trainable_params / all_param
    
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")
    logger.info(f"Percentage trainable: {pct_trainable:.4f}%")
    
    return peft_model
