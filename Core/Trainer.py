"""
Vietnamese Sign Language Detection Training Module
Optimized for CPU training with limited data
"""

import os
import numpy as np
import json
import pickle
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import warnings

# Machine Learning imports
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Deep Learning imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from colorama import Fore, Style, init

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

init(autoreset=True)

class SignLanguageTrainer:
    """
    Optimized trainer for sign language detection with limited data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data paths
        self.data_path = config.get('data_path', 'Data')
        self.models_path = config.get('models_path', 'Models')
        self.logs_path = config.get('logs_path', 'Logs')
        
        # Training parameters
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.n_splits = config.get('n_splits', 5)  # For cross-validation
        
        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        
        # Results tracking
        self.training_history = []
        self.cv_scores = []
        
        # Create directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories"""
        for path in [self.data_path, self.models_path, self.logs_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess training data"""
        try:
            # Load keypoints data
            keypoints_file = Path(self.data_path) / 'keypoints.npy'
            labels_file = Path(self.data_path) / 'labels.npy'
            
            if not keypoints_file.exists() or not labels_file.exists():
                raise FileNotFoundError("Training data files not found")
            
            X = np.load(keypoints_file)
            y = np.load(labels_file)
            
            self.logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
            self.logger.info(f"Classes: {len(np.unique(y))}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for training"""
        # Handle class imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        self.class_weights = dict(zip(np.unique(y), class_weights))
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.logger.info(f"Data preprocessed: {X_scaled.shape}")
        self.logger.info(f"Class weights: {self.class_weights}")
        
        return X_scaled, y_encoded
    
    def create_ml_models(self) -> Dict[str, Any]:
        """Create traditional ML models optimized for small datasets"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            )
        }
        
        return models
    
    def create_dl_model(self, input_shape: int, num_classes: int) -> tf.keras.Model:
        """Create lightweight deep learning model for CPU"""
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_ml_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Train traditional ML models"""
        models = self.create_ml_models()
        scores = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            scores[name] = score
            
            self.logger.info(f"{name} accuracy: {score:.4f}")
            
            # Store model
            self.models[name] = model
        
        return scores
    
    def train_dl_model(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train deep learning model"""
        input_shape = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        model = self.create_dl_model(input_shape, num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                filepath=Path(self.models_path) / 'best_dl_model.keras',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        score = model.evaluate(X_val, y_val, verbose=0)[1]
        self.logger.info(f"Deep Learning accuracy: {score:.4f}")
        
        # Store model
        self.models['deep_learning'] = model
        self.training_history.append(history.history)
        
        return score
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Perform cross-validation for model selection"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        cv_scores = {name: [] for name in ['random_forest', 'gradient_boosting', 'svm']}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.logger.info(f"Cross-validation fold {fold + 1}/{self.n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train models
            models = self.create_ml_models()
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                cv_scores[name].append(score)
        
        # Calculate mean scores
        mean_scores = {name: np.mean(scores) for name, scores in cv_scores.items()}
        self.logger.info("Cross-validation results:")
        for name, score in mean_scores.items():
            self.logger.info(f"{name}: {score:.4f} (+/- {np.std(cv_scores[name]):.4f})")
        
        return cv_scores
    
    def select_best_model(self, scores: Dict[str, float]) -> str:
        """Select the best performing model"""
        best_model_name = max(scores, key=scores.get)
        self.best_score = scores[best_model_name]
        self.best_model = self.models[best_model_name]
        
        self.logger.info(f"Best model: {best_model_name} (accuracy: {self.best_score:.4f})")
        return best_model_name
    
    def save_models(self, best_model_name: str):
        """Save trained models and preprocessing components"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save best model
        if best_model_name == 'deep_learning':
            model_path = Path(self.models_path) / f'best_model_{timestamp}.keras'
            self.best_model.save(model_path)
        else:
            model_path = Path(self.models_path) / f'best_model_{timestamp}.joblib'
            joblib.dump(self.best_model, model_path)
        
        # Save scaler
        scaler_path = Path(self.models_path) / f'scaler_{timestamp}.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        # Save label encoder
        encoder_path = Path(self.models_path) / f'label_encoder_{timestamp}.joblib'
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save action mapping
        action_mapping = {
            'created_date': datetime.now().isoformat(),
            'total_actions': len(self.label_encoder.classes_),
            'actions': {str(i): action for i, action in enumerate(self.label_encoder.classes_)}
        }
        
        mapping_path = Path(self.logs_path) / f'action_mapping_{timestamp}.json'
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(action_mapping, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Models saved with timestamp: {timestamp}")
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'encoder_path': str(encoder_path),
            'mapping_path': str(mapping_path)
        }
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the best model on test set"""
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save evaluation results
        evaluation_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        eval_path = Path(self.logs_path) / f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Test accuracy: {accuracy:.4f}")
        
        return evaluation_results
    
    def plot_training_history(self):
        """Plot training history for deep learning model"""
        if not self.training_history:
            return
        
        history = self.training_history[-1]  # Use last training history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(self.logs_path) / f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history plot saved: {plot_path}")
    
    def train(self) -> Dict[str, Any]:
        """Main training pipeline"""
        self.logger.info("Starting training pipeline...")
        
        try:
            # Load data
            X, y = self.load_data()
            
            # Preprocess data
            X_scaled, y_encoded = self.preprocess_data(X, y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y_encoded
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_train
            )
            
            self.logger.info(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
            
            # Cross-validation for model selection
            cv_scores = self.cross_validate(X_train, y_train)
            
            # Train ML models
            ml_scores = self.train_ml_models(X_train, y_train, X_val, y_val)
            
            # Train DL model
            dl_score = self.train_dl_model(X_train, y_train, X_val, y_val)
            ml_scores['deep_learning'] = dl_score
            
            # Select best model
            best_model_name = self.select_best_model(ml_scores)
            
            # Save models
            saved_paths = self.save_models(best_model_name)
            
            # Evaluate on test set
            evaluation = self.evaluate_model(X_test, y_test)
            
            # Plot training history
            self.plot_training_history()
            
            # Return results
            results = {
                'best_model': best_model_name,
                'best_score': self.best_score,
                'saved_paths': saved_paths,
                'evaluation': evaluation,
                'cv_scores': cv_scores
            }
            
            self.logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
