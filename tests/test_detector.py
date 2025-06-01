"""
Tests for sign language detector.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from src.vsl_detect.core.detector import MediaPipeProcessor, SignLanguageDetector, DetectionResult
from src.vsl_detect.utils.config import Config


class TestMediaPipeProcessor:
    """Test MediaPipe processor."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def processor(self, config):
        """Create MediaPipe processor."""
        with patch('mediapipe.solutions.holistic.Holistic'):
            return MediaPipeProcessor(config)
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert hasattr(processor, 'holistic')
        assert hasattr(processor, 'mp_holistic')
        assert hasattr(processor, 'mp_drawing')
    
    def test_extract_keypoints_empty_results(self, processor):
        """Test keypoint extraction with empty results."""
        # Mock empty results
        mock_results = Mock()
        mock_results.pose_landmarks = None
        mock_results.left_hand_landmarks = None
        mock_results.right_hand_landmarks = None
        
        keypoints = processor.extract_keypoints(mock_results)
        
        # Should return array of zeros with expected shape (126 features)
        assert isinstance(keypoints, np.ndarray)
        assert keypoints.shape == (126,)  # 66 pose + 30 left hand + 30 right hand
        assert np.all(keypoints == 0)
    
    def test_extract_keypoints_with_pose(self, processor):
        """Test keypoint extraction with pose landmarks."""
        # Mock pose landmarks
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.3
        
        mock_results = Mock()
        mock_results.pose_landmarks = Mock()
        mock_results.pose_landmarks.landmark = [mock_landmark] * 33
        mock_results.left_hand_landmarks = None
        mock_results.right_hand_landmarks = None
        
        keypoints = processor.extract_keypoints(mock_results)
        
        assert isinstance(keypoints, np.ndarray)
        assert keypoints.shape == (126,)
        # First 66 values should be non-zero (pose)
        assert np.any(keypoints[:66] != 0)
        # Remaining should be zero (hands)
        assert np.all(keypoints[66:] == 0)


class TestSignLanguageDetector:
    """Test sign language detector."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        # Override model path for testing
        config.model.model_path = "test_model.keras"
        return config
    
    @pytest.fixture
    def mock_model(self):
        """Create mock TensorFlow model."""
        model = Mock()
        # Mock predict method to return dummy predictions
        model.predict.return_value = np.array([[0.1, 0.8, 0.1]])  # High confidence for class 1
        return model
    
    def test_detector_initialization_without_model(self, config):
        """Test detector initialization without model file."""
        with patch('src.vsl_detect.core.detector.MediaPipeProcessor'):
            with pytest.raises(Exception):  # Should raise exception when model file not found
                SignLanguageDetector(config)
    
    @patch('tensorflow.keras.models.load_model')
    @patch('src.vsl_detect.core.detector.MediaPipeProcessor')
    def test_detector_initialization_with_mock_model(self, mock_processor, mock_load_model, config, mock_model):
        """Test detector initialization with mocked model."""
        mock_load_model.return_value = mock_model
        
        # Mock actions loading
        with patch('os.path.exists', return_value=False):
            detector = SignLanguageDetector(config)
        
        assert detector is not None
        assert detector.model is not None
        assert isinstance(detector.actions, list)
    
    @patch('tensorflow.keras.models.load_model')
    @patch('src.vsl_detect.core.detector.MediaPipeProcessor')
    def test_reset_sequence(self, mock_processor, mock_load_model, config, mock_model):
        """Test sequence reset functionality."""
        mock_load_model.return_value = mock_model
        
        with patch('os.path.exists', return_value=False):
            detector = SignLanguageDetector(config)
        
        # Add some dummy data to sequence
        detector.sequence.append(np.zeros(126))
        detector.sequence.append(np.ones(126))
        detector.last_prediction = "test"
        detector.consecutive_count = 5
        
        # Reset
        detector.reset_sequence()
        
        assert len(detector.sequence) == 0
        assert len(detector.predictions) == 0
        assert detector.last_prediction is None
        assert detector.consecutive_count == 0
    
    @patch('tensorflow.keras.models.load_model')
    @patch('src.vsl_detect.core.detector.MediaPipeProcessor')
    def test_process_frame_without_full_sequence(self, mock_processor, mock_load_model, config, mock_model):
        """Test frame processing without full sequence."""
        mock_load_model.return_value = mock_model
        
        # Mock processor
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        mock_processor_instance.process_frame.return_value = (np.zeros((480, 640, 3)), Mock())
        mock_processor_instance.extract_keypoints.return_value = np.zeros(126)
        mock_processor_instance.draw_landmarks.return_value = np.zeros((480, 640, 3))
        
        with patch('os.path.exists', return_value=False):
            detector = SignLanguageDetector(config)
        
        # Process frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.process_frame(frame)
        
        assert isinstance(result, DetectionResult)
        assert result.text == ""  # No text when sequence not full
        assert result.confidence == 0.0


class TestDetectionResult:
    """Test detection result dataclass."""
    
    def test_detection_result_creation(self):
        """Test detection result creation."""
        result = DetectionResult(
            text="hello",
            confidence=0.85,
            landmarks={"pose": "data"},
            timestamp=1234567890.0
        )
        
        assert result.text == "hello"
        assert result.confidence == 0.85
        assert result.landmarks == {"pose": "data"}
        assert result.timestamp == 1234567890.0
    
    def test_detection_result_defaults(self):
        """Test detection result with default values."""
        result = DetectionResult(text="test", confidence=0.5)
        
        assert result.text == "test"
        assert result.confidence == 0.5
        assert result.landmarks is None
        assert result.timestamp is None


if __name__ == "__main__":
    pytest.main([__file__])