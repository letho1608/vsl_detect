"""
Configuration management for Vietnamese Sign Language Detection System.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the ML model."""
    model_path: str = "Models/final_model.keras"
    sequence_length: int = 60
    prediction_threshold: float = 0.7
    min_consecutive_predictions: int = 3
    prediction_smoothing_window: int = 5


@dataclass
class MediaPipeConfig:
    """Configuration for MediaPipe processing."""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    model_complexity: int = 1


@dataclass
class CameraConfig:
    """Configuration for camera capture."""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    flip_horizontal: bool = True


@dataclass
class AudioConfig:
    """Configuration for audio/TTS."""
    language: str = "vi"
    auto_speak: bool = False
    auto_speak_threshold: int = 5
    voice_dir: str = "Voice"
    temp_audio_cleanup: bool = True


@dataclass
class UIConfig:
    """Configuration for the user interface."""
    window_title: str = "Hệ Thống Nhận Dạng Ngôn Ngữ Ký Hiệu"
    window_width: int = 1200
    window_height: int = 800
    show_landmarks: bool = True
    show_confidence: bool = True
    theme: str = "default"


@dataclass
class DataConfig:
    """Configuration for data handling."""
    dataset_dir: str = "Dataset"
    data_dir: str = "Data"
    models_dir: str = "Models"
    logs_dir: str = "Logs"
    checkpoints_dir: str = "Models/checkpoints"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    log_file: str = "Logs/app.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()
        
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'mediapipe' in data:
            config.mediapipe = MediaPipeConfig(**data['mediapipe'])
        if 'camera' in data:
            config.camera = CameraConfig(**data['camera'])
        if 'audio' in data:
            config.audio = AudioConfig(**data['audio'])
        if 'ui' in data:
            config.ui = UIConfig(**data['ui'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'mediapipe': self.mediapipe.__dict__,
            'camera': self.camera.__dict__,
            'audio': self.audio.__dict__,
            'ui': self.ui.__dict__,
            'data': self.data.__dict__,
            'logging': self.logging.__dict__,
        }
    
    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
            elif config_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def get_absolute_path(self, relative_path: str) -> str:
        """Convert relative path to absolute path."""
        if os.path.isabs(relative_path):
            return relative_path
        
        # Get the project root directory
        current_dir = Path(__file__).parent.parent.parent.parent
        return str(current_dir / relative_path)
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.data.dataset_dir,
            self.data.data_dir,
            self.data.models_dir,
            self.data.logs_dir,
            self.data.checkpoints_dir,
            self.audio.voice_dir,
            os.path.dirname(self.logging.log_file),
        ]
        
        for directory in directories:
            abs_path = self.get_absolute_path(directory)
            Path(abs_path).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration with fallback to default."""
    if config_path is None:
        # Try to find config file in standard locations
        config_locations = [
            "configs/config.yaml",
            "configs/config.yml", 
            "configs/config.json",
            "config.yaml",
            "config.yml",
            "config.json",
        ]
        
        for location in config_locations:
            if os.path.exists(location):
                config_path = location
                break
    
    if config_path and os.path.exists(config_path):
        try:
            return Config.from_file(config_path)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration.")
    
    # Return default configuration
    config = Config()
    config.ensure_directories()
    return config