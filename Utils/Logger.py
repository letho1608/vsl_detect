"""
Logging Module for Vietnamese Sign Language Detection
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
import sys
from datetime import datetime

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration
    
    Args:
        config: Logging configuration dictionary
    """
    # Create logs directory
    log_file = config.get('log_file', 'Logs/app.log')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.get('level', 'INFO')))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Console handler
    if config.get('console_handler', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.get('file_handler', True):
        # Use RotatingFileHandler for log rotation
        max_file_size = config.get('max_file_size', 10485760)  # 10MB
        backup_count = config.get('backup_count', 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info(f"Log file: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    return wrapper

class PerformanceLogger:
    """Performance logging utility"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"performance.{name}")
        self.start_time = None
    
    def start(self):
        """Start timing"""
        self.start_time = datetime.now()
        self.logger.debug(f"Started {self.name}")
    
    def end(self):
        """End timing and log duration"""
        if self.start_time:
            duration = datetime.now() - self.start_time
            self.logger.info(f"{self.name} completed in {duration.total_seconds():.3f}s")
            self.start_time = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

class TrainingLogger:
    """Specialized logger for training operations"""
    
    def __init__(self, log_dir: str = "Logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training-specific logger
        self.logger = logging.getLogger("training")
        self.logger.setLevel(logging.INFO)
        
        # Create training log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        training_log_file = self.log_dir / f"training_{timestamp}.log"
        
        # Add file handler for training logs
        file_handler = logging.FileHandler(training_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, 
                  val_loss: float = None, val_acc: float = None):
        """Log training epoch information"""
        if val_loss is not None and val_acc is not None:
            self.logger.info(
                f"Epoch {epoch:3d} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        else:
            self.logger.info(
                f"Epoch {epoch:3d} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics"""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Evaluation Metrics: {metrics_str}")
    
    def log_model_save(self, model_path: str):
        """Log model save operation"""
        self.logger.info(f"Model saved to: {model_path}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log training errors"""
        self.logger.error(f"Training error in {context}: {error}")

class DetectionLogger:
    """Specialized logger for detection operations"""
    
    def __init__(self, log_dir: str = "Logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create detection-specific logger
        self.logger = logging.getLogger("detection")
        self.logger.setLevel(logging.INFO)
        
        # Create detection log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detection_log_file = self.log_dir / f"detection_{timestamp}.log"
        
        # Add file handler for detection logs
        file_handler = logging.FileHandler(detection_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def log_detection(self, action: str, confidence: float, frame_count: int = None):
        """Log detection results"""
        if frame_count is not None:
            self.logger.info(f"Frame {frame_count}: Detected '{action}' with confidence {confidence:.3f}")
        else:
            self.logger.info(f"Detected '{action}' with confidence {confidence:.3f}")
    
    def log_camera_event(self, event: str, details: str = ""):
        """Log camera-related events"""
        if details:
            self.logger.info(f"Camera {event}: {details}")
        else:
            self.logger.info(f"Camera {event}")
    
    def log_model_load(self, model_path: str, success: bool):
        """Log model loading results"""
        if success:
            self.logger.info(f"Model loaded successfully: {model_path}")
        else:
            self.logger.error(f"Failed to load model: {model_path}")
    
    def log_performance(self, fps: float, processing_time: float):
        """Log performance metrics"""
        self.logger.debug(f"Performance - FPS: {fps:.1f}, Processing time: {processing_time:.3f}s")

def create_logger_config(log_level: str = "INFO", 
                        log_file: str = "Logs/app.log",
                        max_file_size: int = 10485760,
                        backup_count: int = 5) -> Dict[str, Any]:
    """
    Create a standard logging configuration
    
    Args:
        log_level: Logging level
        log_file: Log file path
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        
    Returns:
        Logging configuration dictionary
    """
    return {
        'level': log_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_handler': True,
        'console_handler': True,
        'log_file': log_file,
        'max_file_size': max_file_size,
        'backup_count': backup_count
    }
