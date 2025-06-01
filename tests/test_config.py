"""
Tests for configuration system.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.vsl_detect.utils.config import Config, load_config


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()
        
        # Check default values
        assert config.model.sequence_length == 60
        assert config.model.prediction_threshold == 0.7
        assert config.camera.camera_index == 0
        assert config.audio.language == "vi"
        assert config.ui.window_width == 1200
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "camera" in config_dict
        assert "audio" in config_dict
        assert "ui" in config_dict
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        data = {
            "model": {
                "sequence_length": 120,
                "prediction_threshold": 0.8
            },
            "camera": {
                "camera_index": 1,
                "frame_width": 1280
            }
        }
        
        config = Config.from_dict(data)
        
        assert config.model.sequence_length == 120
        assert config.model.prediction_threshold == 0.8
        assert config.camera.camera_index == 1
        assert config.camera.frame_width == 1280
    
    def test_config_save_load_yaml(self):
        """Test saving and loading YAML configuration."""
        config = Config()
        config.model.sequence_length = 90
        config.camera.frame_width = 1920
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save config
            config.save(temp_path)
            assert os.path.exists(temp_path)
            
            # Load config
            loaded_config = Config.from_file(temp_path)
            assert loaded_config.model.sequence_length == 90
            assert loaded_config.camera.frame_width == 1920
            
        finally:
            os.unlink(temp_path)
    
    def test_config_save_load_json(self):
        """Test saving and loading JSON configuration."""
        config = Config()
        config.audio.auto_speak = True
        config.ui.theme = "dark"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save config
            config.save(temp_path)
            assert os.path.exists(temp_path)
            
            # Load config
            loaded_config = Config.from_file(temp_path)
            assert loaded_config.audio.auto_speak == True
            assert loaded_config.ui.theme == "dark"
            
        finally:
            os.unlink(temp_path)
    
    def test_absolute_path(self):
        """Test absolute path conversion."""
        config = Config()
        
        # Test relative path
        rel_path = "Models/test_model.keras"
        abs_path = config.get_absolute_path(rel_path)
        assert os.path.isabs(abs_path)
        assert rel_path in abs_path
        
        # Test absolute path (should remain unchanged)
        if os.name == 'nt':  # Windows
            abs_input = "C:\\test\\path"
        else:  # Unix-like
            abs_input = "/test/path"
        
        result = config.get_absolute_path(abs_input)
        assert result == abs_input
    
    def test_load_config_fallback(self):
        """Test config loading with fallback to default."""
        # Test with non-existent file
        config = load_config("non_existent_config.yaml")
        assert isinstance(config, Config)
        assert config.model.sequence_length == 60  # Default value


if __name__ == "__main__":
    pytest.main([__file__])