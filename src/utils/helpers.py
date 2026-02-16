"""
Shared utility functions: config loading, logging setup, reproducibility.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml
from loguru import logger


def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {path}")
    return config


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.debug(f"Random seed set to {seed}")


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """Configure loguru to write to console and a rotating log file."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / "mp_advisor_{time}.log",
        rotation="10 MB",
        retention="30 days",
        level=level,
    )


def ensure_dirs(config: dict) -> None:
    """Create all output directories specified in config.paths."""
    for key, path in config.get("paths", {}).items():
        Path(path).mkdir(parents=True, exist_ok=True)
