"""
Logging configuration for Vietnamese Sign Language Detection System.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from .config import LoggingConfig


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class VSLLogger:
    """Custom logger for VSL Detection System."""
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def configure(cls, config: LoggingConfig) -> None:
        """Configure the logging system."""
        if cls._configured:
            return
            
        # Create logs directory if it doesn't exist
        log_file_path = Path(config.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if config.console_handler:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, config.level.upper()))
            
            # Use colored formatter for console
            console_formatter = ColoredFormatter(config.format)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if config.file_handler:
            file_handler = logging.handlers.RotatingFileHandler(
                config.log_file,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, config.level.upper()))
            
            # Use standard formatter for file
            file_formatter = logging.Formatter(config.format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance."""
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> logging.Logger:
    """Get a logger instance with optional configuration."""
    if config and not VSLLogger._configured:
        VSLLogger.configure(config)
    elif not VSLLogger._configured:
        # Use default configuration
        from .config import Config
        default_config = Config()
        VSLLogger.configure(default_config.logging)
    
    return VSLLogger.get_logger(name)


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Setup logging configuration."""
    if config is None:
        from .config import Config
        config = Config().logging
    
    VSLLogger.configure(config)


# Convenience functions for different log levels
def log_info(message: str, logger_name: str = "vsl_detect") -> None:
    """Log info message."""
    logger = get_logger(logger_name)
    logger.info(message)


def log_error(message: str, logger_name: str = "vsl_detect", exc_info: bool = False) -> None:
    """Log error message."""
    logger = get_logger(logger_name)
    logger.error(message, exc_info=exc_info)


def log_warning(message: str, logger_name: str = "vsl_detect") -> None:
    """Log warning message."""
    logger = get_logger(logger_name)
    logger.warning(message)


def log_debug(message: str, logger_name: str = "vsl_detect") -> None:
    """Log debug message."""
    logger = get_logger(logger_name)
    logger.debug(message)


def log_critical(message: str, logger_name: str = "vsl_detect") -> None:
    """Log critical message."""
    logger = get_logger(logger_name)
    logger.critical(message)


# Context manager for performance logging
class PerformanceLogger:
    """Context manager for logging execution time."""
    
    def __init__(self, operation_name: str, logger_name: str = "vsl_detect.performance"):
        self.operation_name = operation_name
        self.logger = get_logger(logger_name)
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        end_time = time.time()
        duration = end_time - self.start_time
        
        if exc_type is None:
            self.logger.debug(f"Completed {self.operation_name} in {duration:.3f}s")
        else:
            self.logger.error(f"Failed {self.operation_name} after {duration:.3f}s: {exc_val}")


# Decorator for function performance logging
def log_performance(operation_name: Optional[str] = None):
    """Decorator to log function execution time."""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            with PerformanceLogger(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Exception logging decorator
def log_exceptions(logger_name: str = "vsl_detect.exceptions", reraise: bool = True):
    """Decorator to log exceptions."""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger(logger_name)
                logger.error(
                    f"Exception in {func.__module__}.{func.__name__}: {e}",
                    exc_info=True
                )
                if reraise:
                    raise
                return None
        return wrapper
    return decorator