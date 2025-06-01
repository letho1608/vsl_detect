"""
Main entry point for Vietnamese Sign Language Detection System.
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for development
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer

from vsl_detect.utils.config import load_config, Config
from vsl_detect.utils.logger import setup_logging, get_logger
from vsl_detect.ui.main_window import SignLanguageApp


def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    logger = get_logger(__name__)
    
    missing_deps = []
    
    try:
        import cv2
        logger.debug("OpenCV available")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import mediapipe
        logger.debug("MediaPipe available")
    except ImportError:
        missing_deps.append("mediapipe")
    
    try:
        import tensorflow
        logger.debug("TensorFlow available")
    except ImportError:
        missing_deps.append("tensorflow")
    
    try:
        import numpy
        logger.debug("NumPy available")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import gtts
        logger.debug("gTTS available")
    except ImportError:
        missing_deps.append("gtts")
    
    try:
        import pygame
        logger.debug("Pygame available")
    except ImportError:
        missing_deps.append("pygame")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        return False
    
    logger.info("All dependencies are available")
    return True


def check_model_files(config: Config) -> bool:
    """Check if required model files exist."""
    logger = get_logger(__name__)
    
    model_path = config.get_absolute_path(config.model.model_path)
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    logger.info(f"Model file found: {model_path}")
    return True


def setup_environment(config: Config) -> None:
    """Setup environment and create required directories."""
    logger = get_logger(__name__)
    
    try:
        # Ensure all required directories exist
        config.ensure_directories()
        logger.info("Environment setup completed")
        
        # Set environment variables for better performance
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available
        
    except Exception as e:
        logger.error(f"Failed to setup environment: {e}")
        raise


def main(config_path: Optional[str] = None) -> int:
    """
    Main application entry point.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1
    
    # Setup logging
    try:
        setup_logging(config.logging)
        logger = get_logger(__name__)
        logger.info("Starting Vietnamese Sign Language Detection System")
        logger.info(f"Using configuration: {config_path or 'default'}")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing required dependencies")
        return 1
    
    # Setup environment
    try:
        setup_environment(config)
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return 1
    
    # Check model files
    if not check_model_files(config):
        logger.error("Required model files not found")
        return 1
    
    # Create Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("VSL Detect")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Vietnamese Sign Language Team")
    
    try:
        # Create main window
        logger.info("Creating main application window")
        main_window = SignLanguageApp(config)
        main_window.show()
        
        # Setup graceful shutdown
        def cleanup():
            logger.info("Shutting down application")
            main_window.cleanup()
        
        app.aboutToQuit.connect(cleanup)
        
        # Start event loop
        logger.info("Starting application event loop")
        exit_code = app.exec_()
        
        logger.info(f"Application exited with code: {exit_code}")
        return exit_code
        
    except Exception as e:
        logger.critical(f"Critical error in main application: {e}", exc_info=True)
        
        # Show error dialog if possible
        try:
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setWindowTitle("Lỗi Nghiêm Trọng")
            error_msg.setText("Ứng dụng gặp lỗi nghiêm trọng và phải đóng.")
            error_msg.setDetailedText(str(e))
            error_msg.exec_()
        except:
            pass  # Qt might not be available
        
        return 1


def cli_entry_point():
    """CLI entry point for console scripts."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Vietnamese Sign Language Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="VSL Detect 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Override debug level if specified
    if args.debug:
        if args.config:
            config = load_config(args.config)
        else:
            config = load_config()
        config.logging.level = "DEBUG"
        
        # Save temporary config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save(f.name)
            args.config = f.name
    
    exit_code = main(args.config)
    sys.exit(exit_code)


if __name__ == "__main__":
    cli_entry_point()