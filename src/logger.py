import logging
import sys
from typing import Optional

def setup_logger(name: str = "DocForge", level: int = logging.INFO) -> logging.Logger:
    """Set up and configure the logger.

    Args:
        name: Name of the logger.
        level: Logging level (e.g. logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, don't add them again
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid double logging
    logger.propagate = False
    
    return logger

def get_logger(name: str = "DocForge") -> logging.Logger:
    """Retrieve the logger for the project.

    Args:
        name: Name of the logger.

    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger(name)

# Initialize project-wide root logger
setup_logger()
