"""
Core sign language detection engine.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from ..utils.config import Config
from ..utils.logger import get_logger, log_performance


@dataclass
class DetectionResult:
    """Result of sign language detection."""
    text: str
    confidence: float
    landmarks: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


class MediaPipeProcessor:
    """MediaPipe landmark extraction processor."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=config.mediapipe.min_detection_confidence,
            min_tracking_confidence=config.mediapipe.min_tracking_confidence,
            static_image_mode=config.mediapipe.static_image_mode,
            model_complexity=config.mediapipe.model_complexity
        )
        
        self.logger.info("MediaPipe processor initialized")
    
    @log_performance("mediapipe_detection")
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Process frame with MediaPipe.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, results)
        """
        # Flip frame horizontally if configured
        if self.config.camera.flip_horizontal:
            frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = self.holistic.process(frame_rgb)
        
        # Convert back to BGR
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr, results
    
    def extract_keypoints(self, results: Any) -> np.ndarray:
        """
        Extract keypoints from MediaPipe results.
        
        Args:
            results: MediaPipe results
            
        Returns:
            Keypoints array (126 features)
        """
        # Pose keypoints (33 points * 2 coordinates = 66 features)
        pose = np.array([
            [res.x, res.y] for res in results.pose_landmarks.landmark
        ]).flatten() if results.pose_landmarks else np.zeros(33 * 2)
        
        # Hand keypoints (15 important points * 2 coordinates = 30 features each)
        important_hand_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        
        # Left hand
        if results.left_hand_landmarks:
            lh = np.array([
                [results.left_hand_landmarks.landmark[idx].x,
                 results.left_hand_landmarks.landmark[idx].y]
                for idx in important_hand_indices
            ]).flatten()
        else:
            lh = np.zeros(30)
        
        # Right hand
        if results.right_hand_landmarks:
            rh = np.array([
                [results.right_hand_landmarks.landmark[idx].x,
                 results.right_hand_landmarks.landmark[idx].y]
                for idx in important_hand_indices
            ]).flatten()
        else:
            rh = np.zeros(30)
        
        return np.concatenate([pose, lh, rh])
    
    def draw_landmarks(self, image: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw landmarks on image.
        
        Args:
            image: Input image
            results: MediaPipe results
            
        Returns:
            Image with landmarks drawn
        """
        # Face landmarks
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )
        
        # Pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
        
        # Hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        return image
    
    def cleanup(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'holistic'):
            self.holistic.close()
        self.logger.info("MediaPipe processor cleaned up")


class SignLanguageDetector:
    """Main sign language detection engine."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.processor = MediaPipeProcessor(config)
        self.model = None
        self.actions = []
        self.action_mapping = {}
        
        # Detection state
        self.sequence = deque(maxlen=config.model.sequence_length)
        self.predictions = deque(maxlen=config.model.prediction_smoothing_window)
        self.last_prediction = None
        self.consecutive_count = 0
        
        # Load model and actions
        self._load_model()
        self._load_actions()
        
        self.logger.info("Sign language detector initialized")
    
    def _load_model(self) -> None:
        """Load the trained model."""
        try:
            model_path = self.config.get_absolute_path(self.config.model.model_path)
            self.model = tf.keras.models.load_model(model_path)
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_actions(self) -> None:
        """Load action mappings."""
        try:
            import json
            import os
            
            # Try to load from mapping file
            mapping_file = self.config.get_absolute_path("Logs/action_mapping.json")
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                    if isinstance(mapping_data, dict) and 'actions' in mapping_data:
                        self.action_mapping = mapping_data['actions']
                        self.actions = sorted(list(self.action_mapping.keys()))
                        self.logger.info("Action mapping loaded successfully")
                        return
            
            # Fallback: load from data directory
            data_dir = self.config.get_absolute_path(self.config.data.data_dir)
            if os.path.exists(data_dir):
                self.actions = sorted(os.listdir(data_dir))
                self.action_mapping = {action: action for action in self.actions}
                self.logger.info(f"Loaded {len(self.actions)} actions from data directory")
            else:
                self.logger.warning("No actions found - detector will not work properly")
                
        except Exception as e:
            self.logger.error(f"Failed to load actions: {e}")
            self.actions = []
            self.action_mapping = {}
    
    @log_performance("frame_detection")
    def process_frame(self, frame: np.ndarray, draw_landmarks: bool = True) -> DetectionResult:
        """
        Process a single frame for sign language detection.
        
        Args:
            frame: Input frame
            draw_landmarks: Whether to draw landmarks on frame
            
        Returns:
            DetectionResult with text and confidence
        """
        # Process with MediaPipe
        processed_frame, results = self.processor.process_frame(frame)
        
        # Draw landmarks if requested
        if draw_landmarks:
            processed_frame = self.processor.draw_landmarks(processed_frame, results)
        
        # Extract keypoints
        keypoints = self.processor.extract_keypoints(results)
        self.sequence.append(keypoints)
        
        # Make prediction if sequence is full
        text = ""
        confidence = 0.0
        
        if len(self.sequence) == self.config.model.sequence_length and self.model is not None:
            text, confidence = self._make_prediction()
        
        return DetectionResult(
            text=text,
            confidence=confidence,
            landmarks=results,
            timestamp=None  # Could add timestamp if needed
        )
    
    def _make_prediction(self) -> Tuple[str, float]:
        """
        Make prediction from current sequence.
        
        Returns:
            Tuple of (predicted_text, confidence)
        """
        try:
            # Prepare input data
            input_data = np.expand_dims(list(self.sequence), axis=0)
            
            # Make prediction
            pred = self.model.predict(input_data, verbose=0)[0]
            self.predictions.append(pred)
            
            # Average predictions for stability
            avg_pred = np.mean(list(self.predictions), axis=0)
            max_confidence = float(np.max(avg_pred))
            
            # Check if confidence is above threshold
            if max_confidence > self.config.model.prediction_threshold:
                pred_idx = int(np.argmax(avg_pred))
                
                if pred_idx < len(self.actions):
                    predicted_action = self.actions[pred_idx]
                    
                    # Check for consecutive predictions
                    if predicted_action == self.last_prediction:
                        self.consecutive_count += 1
                    else:
                        self.consecutive_count = 1
                        self.last_prediction = predicted_action
                    
                    # Return prediction if we have enough consecutive matches
                    if self.consecutive_count >= self.config.model.min_consecutive_predictions:
                        # Get original text from mapping
                        original_text = self.action_mapping.get(predicted_action, predicted_action)
                        return original_text, max_confidence
            
            return "", 0.0
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return "", 0.0
    
    def reset_sequence(self) -> None:
        """Reset the detection sequence."""
        self.sequence.clear()
        self.predictions.clear()
        self.last_prediction = None
        self.consecutive_count = 0
        self.logger.debug("Detection sequence reset")
    
    def get_actions(self) -> List[str]:
        """Get list of available actions."""
        return list(self.action_mapping.values())
    
    def cleanup(self) -> None:
        """Cleanup detector resources."""
        if self.processor:
            self.processor.cleanup()
        self.logger.info("Sign language detector cleaned up")