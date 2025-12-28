# utils/logger.py
"""
Logging utilities for the Harvard Research Paper Publication Crew.
"""

import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str, 
    log_file: Optional[str] = None, 
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_execution_time(func):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger = logging.getLogger(func.__module__)
        
        try:
            logger.info(f"Starting execution of {func.__name__}")
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            logger.info(f"Completed {func.__name__} in {execution_time}")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            logger.error(f"Error in {func.__name__} after {execution_time}: {str(e)}")
            raise
    
    return wrapper
