"""
Utilities: Logging setup, common functions used across modules.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

from src import config


# =============================================================================
# LOGGING SETUP
# =============================================================================

_logger_initialized = False


def setup_logging(
    name: str = "vinbigdata",
    log_dir: str = None,
    log_file: str = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Setup logging with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (default: config.OUTPUT_DIR/logs)
        log_file: Log filename (default: auto-generated with timestamp)
        level: Logging level
        console: Whether to also log to console
    
    Returns:
        Configured logger
    """
    global _logger_initialized
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers on repeated calls
    if _logger_initialized:
        return logger
    
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Setup log directory
    if log_dir is None:
        log_dir = os.path.join(config.OUTPUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"run_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    _logger_initialized = True
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger


def get_logger(name: str = "vinbigdata") -> logging.Logger:
    """Get or create logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logging(name)
    return logger


# =============================================================================
# DEVICE & ENVIRONMENT
# =============================================================================

def get_device() -> torch.device:
    """Get best available device."""
    return config.DEVICE


def log_system_info(logger: logging.Logger = None):
    """Log system and environment info."""
    if logger is None:
        logger = get_logger()
    
    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    logger.info(f"Device: {config.DEVICE}")
    logger.info("=" * 60)


# =============================================================================
# COMMON UTILITIES
# =============================================================================

def ensure_dir(path: str) -> str:
    """Create directory if not exists, return path."""
    os.makedirs(path, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
    }


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# =============================================================================
# NUMPY / TENSOR UTILITIES
# =============================================================================

def to_numpy(tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_trained_models(fold: int = 0):
    """Find all trained models for a fold."""
    available = []
    for model_name in config.AVAILABLE_MODELS:
        path = config.get_checkpoint_path(model_name, fold)
        if os.path.exists(path):
            available.append(model_name)
    return available