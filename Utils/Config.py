"""
Configuration Management Module
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_path: str = "Models/final_model.keras"
    sequence_length: int = 60
    prediction_threshold: float = 0.7
    min_consecutive_predictions: int = 3
    prediction_smoothing_window: int = 5

@dataclass
class MediaPipeConfig:
    """MediaPipe configuration settings"""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    model_complexity: int = 1

@dataclass
class CameraConfig:
    """Camera configuration settings"""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    flip_horizontal: bool = True

@dataclass
class AudioConfig:
    """Audio/TTS configuration settings"""
    language: str = "vi"
    auto_speak: bool = False
    auto_speak_threshold: int = 5
    voice_dir: str = "Voice"
    temp_audio_cleanup: bool = True

@dataclass
class UIConfig:
    """UI configuration settings"""
    window_title: str = "Hệ Thống Nhận Dạng Ngôn Ngữ Ký Hiệu"
    window_width: int = 1200
    window_height: int = 800
    show_landmarks: bool = True
    show_confidence: bool = True
    theme: str = "default"

@dataclass
class DataConfig:
    """Data paths configuration"""
    dataset_dir: str = "Dataset"
    data_dir: str = "Data"
    models_dir: str = "Models"
    logs_dir: str = "Logs"
    checkpoints_dir: str = "Models/checkpoints"

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    log_file: str = "Logs/app.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    augmentation_factor: int = 3
    noise_factor: float = 0.01
    rotation_range: int = 5

class ConfigManager:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "Configs/config.yaml"
        self.logger = logging.getLogger(__name__)
        
        # Initialize default configurations
        self.model = ModelConfig()
        self.mediapipe = MediaPipeConfig()
        self.camera = CameraConfig()
        self.audio = AudioConfig()
        self.ui = UIConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        self.training = TrainingConfig()
        
        # Load configuration
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # Update configurations
                self._update_configs(config_data)
                self.logger.info(f"Configuration loaded from {self.config_path}")
                return True
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                self.save_config()  # Save default config
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'model': asdict(self.model),
                'mediapipe': asdict(self.mediapipe),
                'camera': asdict(self.camera),
                'audio': asdict(self.audio),
                'ui': asdict(self.ui),
                'data': asdict(self.data),
                'logging': asdict(self.logging),
                'training': asdict(self.training)
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _update_configs(self, config_data: Dict[str, Any]):
        """Update configuration objects from dictionary"""
        if 'model' in config_data:
            for key, value in config_data['model'].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        if 'mediapipe' in config_data:
            for key, value in config_data['mediapipe'].items():
                if hasattr(self.mediapipe, key):
                    setattr(self.mediapipe, key, value)
        
        if 'camera' in config_data:
            for key, value in config_data['camera'].items():
                if hasattr(self.camera, key):
                    setattr(self.camera, key, value)
        
        if 'audio' in config_data:
            for key, value in config_data['audio'].items():
                if hasattr(self.audio, key):
                    setattr(self.audio, key, value)
        
        if 'ui' in config_data:
            for key, value in config_data['ui'].items():
                if hasattr(self.ui, key):
                    setattr(self.ui, key, value)
        
        if 'data' in config_data:
            for key, value in config_data['data'].items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        
        if 'logging' in config_data:
            for key, value in config_data['logging'].items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
        
        if 'training' in config_data:
            for key, value in config_data['training'].items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)
    
    def get_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            'model': asdict(self.model),
            'mediapipe': asdict(self.mediapipe),
            'camera': asdict(self.camera),
            'audio': asdict(self.audio),
            'ui': asdict(self.ui),
            'data': asdict(self.data),
            'logging': asdict(self.logging),
            'training': asdict(self.training)
        }
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """Update a specific configuration value"""
        try:
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                    self.save_config()
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data.dataset_dir,
            self.data.data_dir,
            self.data.models_dir,
            self.data.logs_dir,
            self.data.checkpoints_dir,
            self.audio.voice_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration settings"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate model path
        if not Path(self.model.model_path).parent.exists():
            validation_results['warnings'].append(f"Model directory does not exist: {self.model.model_path}")
        
        # Validate camera settings
        if self.camera.fps < 1 or self.camera.fps > 60:
            validation_results['errors'].append("Camera FPS must be between 1 and 60")
        
        if self.camera.frame_width < 320 or self.camera.frame_height < 240:
            validation_results['warnings'].append("Camera resolution is very low")
        
        # Validate prediction threshold
        if self.model.prediction_threshold < 0.0 or self.model.prediction_threshold > 1.0:
            validation_results['errors'].append("Prediction threshold must be between 0.0 and 1.0")
        
        # Check for errors
        if validation_results['errors']:
            validation_results['valid'] = False
        
        return validation_results

def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """Load configuration from file"""
    return ConfigManager(config_path)

def create_default_config(config_path: str = "Configs/config.yaml") -> bool:
    """Create default configuration file"""
    try:
        config_manager = ConfigManager(config_path)
        return config_manager.save_config()
    except Exception as e:
        logging.error(f"Failed to create default config: {e}")
        return False
