"""
Logging utilities for consistent logging across the application.
"""

import logging
import sys
from typing import Optional
from src.config import Config


def setup_logger(
    name: str,
    level: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (defaults to Config.LOG_LEVEL)
        format_string: Custom format string (defaults to Config.LOG_FORMAT)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    level = level or Config.LOG_LEVEL
    format_string = format_string or Config.LOG_FORMAT
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance (creates one if it doesn't exist).
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        return setup_logger(name)
    
    return logger

