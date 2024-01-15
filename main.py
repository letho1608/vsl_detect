#!/usr/bin/env python3
"""
Vietnamese Sign Language Detection System
Main Application Entry Point
"""

import sys
import os
from pathlib import Path
from typing import Optional
import argparse
import logging

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

from Utils.Config import ConfigManager, load_config
from Utils.Logger import setup_logging, get_logger
from UI.MainWindow import SignLanguageApp

def check_dependencies() -> bool:
    """Check if all required dependencies are available"""
    logger = get_logger(__name__)
    
    required_packages = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('tensorflow', 'tensorflow'),
        ('numpy', 'numpy'),
        ('PyQt5', 'PyQt5'),
        ('yaml', 'PyYAML'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
        ('colorama', 'colorama'),
        ('tabulate', 'tabulate'),
        ('joblib', 'joblib')
    ]
    
    missing_packages = []
    
    for package_name, pip_name in required_packages:
        try:
            __import__(package_name)
            logger.debug(f"✓ {package_name} available")
        except ImportError:
            missing_packages.append(pip_name)
            logger.warning(f"✗ {package_name} not found")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("All dependencies are available")
    return True

def setup_environment() -> bool:
    """Setup application environment"""
    logger = get_logger(__name__)
    
    try:
        # Create necessary directories
        directories = ['Data', 'Models', 'Logs', 'Configs', 'Dataset']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        # Set environment variables for better performance
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
        
        logger.info("Environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return False

def run_gui(config_path: Optional[str] = None) -> int:
    """Run the GUI application"""
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config_manager = load_config(config_path)
        config = config_manager.get_dict()
        
        # Setup logging
        setup_logging(config['logging'])
        logger.info("Starting Vietnamese Sign Language Detection System")
        
        # Check dependencies
        if not check_dependencies():
            logger.error("Missing required dependencies")
            return 1
        
        # Setup environment
        if not setup_environment():
            logger.error("Environment setup failed")
            return 1
        
        # Create Qt Application
        app = QApplication(sys.argv)
        app.setApplicationName("VSL Detect")
        app.setApplicationVersion("2.0.0")
        app.setOrganizationName("Vietnamese Sign Language Team")
        
        # Set application style
        app.setStyle('Fusion')
        
        # Create main window
        logger.info("Creating main application window")
        main_window = SignLanguageApp(config)
        main_window.show()
        
        # Setup graceful shutdown
        def cleanup():
            logger.info("Shutting down application")
            main_window.close()
        
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
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setWindowTitle("Lỗi Nghiêm Trọng")
            error_msg.setText("Ứng dụng gặp lỗi nghiêm trọng và phải đóng.")
            error_msg.setDetailedText(str(e))
            error_msg.exec_()
        except:
            pass
        
        return 1

def run_training(config_path: Optional[str] = None) -> int:
    """Run the training module"""
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config_manager = load_config(config_path)
        config = config_manager.get_dict()
        
        # Setup logging
        setup_logging(config['logging'])
        logger.info("Starting training module")
        
        # Import training module
        from Core.Trainer import SignLanguageTrainer
        
        # Create trainer
        trainer = SignLanguageTrainer(config['training'])
        
        # Run training
        results = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model: {results['best_model']}")
        logger.info(f"Best score: {results['best_score']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

def run_data_processing(config_path: Optional[str] = None) -> int:
    """Run the data processing module"""
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config_manager = load_config(config_path)
        config = config_manager.get_dict()
        
        # Setup logging
        setup_logging(config['logging'])
        logger.info("Starting data processing module")
        
        # Import data processor
        from Data.Processor import DataProcessor
        
        # Create processor
        processor = DataProcessor(config['training'])
        
        # Process dataset
        success = processor.process_dataset()
        
        if success:
            # Validate processed data
            validation = processor.validate_data()
            if validation['valid']:
                logger.info("Data processing completed successfully!")
                logger.info(f"Processed {validation['samples']} samples")
                logger.info(f"Features: {validation['features']}")
                logger.info(f"Actions: {validation['actions']}")
                return 0
            else:
                logger.error(f"Data validation failed: {validation.get('error', 'Unknown error')}")
                return 1
        else:
            logger.error("Data processing failed")
            return 1
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}", exc_info=True)
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Vietnamese Sign Language Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run GUI application
  python main.py --gui              # Run GUI application
  python main.py --train            # Run training module
  python main.py --process-data     # Run data processing
  python main.py --config custom.yaml  # Use custom config file
        """
    )
    
    parser.add_argument(
        "--gui", "-g",
        action="store_true",
        help="Run GUI application (default)"
    )
    
    parser.add_argument(
        "--train", "-t",
        action="store_true",
        help="Run training module"
    )
    
    parser.add_argument(
        "--process-data", "-p",
        action="store_true",
        help="Run data processing module"
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
        version="VSL Detect 2.0.0"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.train:
        mode = "train"
    elif args.process_data:
        mode = "process_data"
    else:
        mode = "gui"  # Default mode
    
    # Run appropriate module
    if mode == "gui":
        exit_code = run_gui(args.config)
    elif mode == "train":
        exit_code = run_training(args.config)
    elif mode == "process_data":
        exit_code = run_data_processing(args.config)
    else:
        print(f"Unknown mode: {mode}")
        return 1
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()