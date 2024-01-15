"""
Vietnamese Sign Language Detection Core Module
Optimized for CPU training with limited data
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Configure TensorFlow for CPU optimization
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SignLanguageDetector:
    """
    Optimized sign language detector for CPU with limited training data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=config.get('min_tracking_confidence', 0.5),
            static_image_mode=False,
            model_complexity=1  # Use lighter model for CPU
        )
        
        # Model components
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.action_mapping = {}
        
        # Detection state
        self.sequence = []
        self.predictions = []
        self.current_action = None
        self.confidence_threshold = config.get('prediction_threshold', 0.7)
        self.min_consecutive = config.get('min_consecutive_predictions', 3)
        
        # Performance optimization
        self.frame_skip = config.get('frame_skip', 2)  # Process every nth frame
        self.frame_count = 0
        
    def load_model(self, model_path: str) -> bool:
        """Load trained model with fallback options"""
        try:
            # Try loading TensorFlow model first
            if model_path.endswith('.keras') or model_path.endswith('.h5'):
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.logger.info(f"Loaded TensorFlow model: {model_path}")
                return True
                
            # Try loading scikit-learn model
            elif model_path.endswith('.pkl') or model_path.endswith('.joblib'):
                if model_path.endswith('.joblib'):
                    self.model = joblib.load(model_path)
                else:
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                self.logger.info(f"Loaded scikit-learn model: {model_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            
        return False
    
    def load_scaler(self, scaler_path: str) -> bool:
        """Load feature scaler"""
        try:
            if scaler_path.endswith('.joblib'):
                self.scaler = joblib.load(scaler_path)
            else:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            self.logger.info(f"Loaded scaler: {scaler_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {e}")
            return False
    
    def load_action_mapping(self, mapping_path: str) -> bool:
        """Load action mapping"""
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.action_mapping = json.load(f)
            self.logger.info(f"Loaded action mapping: {mapping_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load action mapping: {e}")
            return False
    
    def extract_keypoints(self, results) -> Optional[np.ndarray]:
        """Extract and validate keypoints from MediaPipe results"""
        try:
            # Extract hand landmarks
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            
            # Extract pose landmarks (upper body only for efficiency)
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:25]]).flatten() if results.pose_landmarks else np.zeros(25*3)
            
            # Combine keypoints
            keypoints = np.concatenate([lh, rh, pose])
            
            # Validate keypoints
            if np.isnan(keypoints).any() or np.isinf(keypoints).any():
                return None
                
            return keypoints
            
        except Exception as e:
            self.logger.debug(f"Keypoint extraction error: {e}")
            return None
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Process frame and extract keypoints"""
        # Skip frames for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return frame, None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process with MediaPipe
        results = self.holistic.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract keypoints
        keypoints = self.extract_keypoints(results)
        
        return image, keypoints
    
    def predict_action(self, keypoints: np.ndarray) -> Tuple[str, float]:
        """Predict action from keypoints"""
        if self.model is None or keypoints is None:
            return "Unknown", 0.0
        
        try:
            # Scale features if scaler is available
            if self.scaler is not None:
                keypoints_scaled = self.scaler.transform(keypoints.reshape(1, -1))
            else:
                keypoints_scaled = keypoints.reshape(1, -1)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                # scikit-learn model
                probabilities = self.model.predict_proba(keypoints_scaled)[0]
                predicted_class = self.model.predict(keypoints_scaled)[0]
                confidence = np.max(probabilities)
            else:
                # TensorFlow model
                prediction = self.model.predict(keypoints_scaled, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
            
            # Map class to action name
            action_name = self.action_mapping.get(str(predicted_class), "Unknown")
            
            return action_name, confidence
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return "Unknown", 0.0
    
    def update_sequence(self, keypoints: np.ndarray):
        """Update sequence for temporal analysis"""
        if keypoints is not None:
            self.sequence.append(keypoints)
            
            # Keep only recent frames
            max_sequence_length = self.config.get('sequence_length', 30)
            if len(self.sequence) > max_sequence_length:
                self.sequence.pop(0)
    
    def get_temporal_prediction(self) -> Tuple[str, float]:
        """Get prediction based on temporal sequence"""
        if len(self.sequence) < 5:  # Need minimum frames
            return "Unknown", 0.0
        
        # Use simple voting mechanism for temporal consistency
        predictions = []
        confidences = []
        
        for keypoints in self.sequence[-10:]:  # Last 10 frames
            action, conf = self.predict_action(keypoints)
            predictions.append(action)
            confidences.append(conf)
        
        # Find most common prediction with high confidence
        from collections import Counter
        pred_counter = Counter(predictions)
        most_common_action = pred_counter.most_common(1)[0][0]
        
        # Calculate average confidence for most common action
        action_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == most_common_action]
        avg_confidence = np.mean(action_confidences) if action_confidences else 0.0
        
        return most_common_action, avg_confidence
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame and return results"""
        # Preprocess frame
        processed_frame, keypoints = self.preprocess_frame(frame)
        
        # Update sequence
        self.update_sequence(keypoints)
        
        # Get prediction
        if keypoints is not None:
            action, confidence = self.get_temporal_prediction()
        else:
            action, confidence = "Unknown", 0.0
        
        # Update current action with confidence threshold
        if confidence > self.confidence_threshold:
            self.current_action = action
        elif confidence < 0.3:  # Low confidence threshold
            self.current_action = None
        
        return {
            'frame': processed_frame,
            'action': self.current_action,
            'confidence': confidence,
            'keypoints': keypoints,
            'sequence_length': len(self.sequence)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.holistic:
            self.holistic.close()
        self.logger.info("Detector cleanup completed")
