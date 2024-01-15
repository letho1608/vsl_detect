"""
Vietnamese Sign Language Data Processing Module
Optimized for limited data scenarios
"""

import cv2
import numpy as np
import os
import mediapipe as mp
import pandas as pd
import json
import unicodedata
import re
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import warnings
from datetime import datetime
from colorama import init, Fore, Style
import multiprocessing
from functools import partial
import requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import csv

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

init(autoreset=True)

class DataProcessor:
    """
    Optimized data processor for sign language detection with limited data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("Processor initialized with dataset: %s", self.dataset_path)
        
        # Paths
        self.dataset_path = config.get('dataset_path', 'Dataset')
        self.data_path = config.get('data_path', 'Data')
        self.logs_path = config.get('logs_path', 'Logs')
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            model_complexity=1  # Lighter model for CPU
        )
        
        # Data augmentation settings
        self.augmentation_factor = config.get('augmentation_factor', 3)
        self.noise_factor = config.get('noise_factor', 0.01)
        self.rotation_range = config.get('rotation_range', 5)
        
        # Progress tracking
        self.stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'total_frames': 0,
            'extracted_frames': 0
        }
        
        # Create directories
        self._create_directories()

    # =============================
    # Video collection integration
    # =============================
    def _normalize_labels_csv(self, csv_path: Path) -> None:
        """Normalize Label.csv to schema with columns: Video, Label

        Supports legacy schema created by Combine.py: [STT, VIDEO, TEXT].
        """
        try:
            if not csv_path.exists():
                return
            df = pd.read_csv(csv_path)
            # If already correct, do nothing
            lowered = [c.lower() for c in df.columns]
            if 'video' in lowered and 'label' in lowered and len(df.columns) >= 2:
                # Ensure exact column names
                colmap: Dict[str, str] = {}
                for c in df.columns:
                    lc = c.lower()
                    if lc == 'video':
                        colmap[c] = 'Video'
                    elif lc == 'label':
                        colmap[c] = 'Label'
                df = df.rename(columns=colmap)
                df[['Video', 'Label']].to_csv(csv_path, index=False, encoding='utf-8')
                return

            # Legacy schema: STT, VIDEO, TEXT
            legacy_candidates = [c.lower() for c in df.columns]
            if 'video' in legacy_candidates or 'text' in legacy_candidates or 'stt' in legacy_candidates:
                # Map VIDEO -> Video, TEXT -> Label
                video_col = None
                label_col = None
                for c in df.columns:
                    lc = c.lower()
                    if lc == 'video':
                        video_col = c
                    if lc == 'text':
                        label_col = c
                if video_col and label_col:
                    out_df = pd.DataFrame({'Video': df[video_col], 'Label': df[label_col]})
                    out_df.to_csv(csv_path, index=False, encoding='utf-8')
        except Exception as e:
            self.logger.warning(f"Could not normalize labels CSV: {e}")

    def _download_single_video(self, output_dir: Path, csv_path: Path, item: Dict[str, Any]) -> None:
        """Download one video from item {'url': str, 'gross': str} and append to Label.csv (Video, Label)"""
        try:
            url: str = item.get('url')
            label_text: str = item.get('gross', '')
            if not url:
                return

            filename = os.path.basename(urlparse(url).path)
            out_path = output_dir / filename
            if out_path.exists():
                return

            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(out_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=f"Downloading {filename}", ncols=100) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Append to CSV with schema Video,Label
            header_needed = not csv_path.exists()
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow(["Video", "Label"])
                writer.writerow([filename, label_text])

        except Exception as e:
            # Cleanup partial
            try:
                if out_path and out_path.exists():
                    out_path.unlink(missing_ok=True)
            except Exception:
                pass
            self.logger.warning(f"Failed downloading {item}: {e}")

    def collect_videos_via_crawler(self, max_workers: int = 3) -> bool:
        """Use Data.Combine crawler (if available) to collect videos and create Label.csv

        Returns True if collection executed (even if zero results), False if crawler unavailable.
        """
        try:
            # Import lazily to avoid hard dependency
            from Data.Combine import crawl_videos
        except Exception as e:
            self.logger.info("Crawler not available (Data.Combine.crawl_videos not found). Skipping auto-collection.")
            return False

        try:
            output_dir = Path(self.dataset_path) / 'Video'
            text_dir = Path(self.dataset_path) / 'Text'
            output_dir.mkdir(parents=True, exist_ok=True)
            text_dir.mkdir(parents=True, exist_ok=True)
            csv_path = text_dir / 'Label.csv'

            self.logger.info("Starting crawler to collect videos...")
            items: List[Dict[str, Any]] = crawl_videos()
            if not items:
                self.logger.warning("Crawler returned no items")
                return True

            # Download in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for item in items:
                    executor.submit(self._download_single_video, output_dir, csv_path, item)

            # Normalize CSV to target schema
            self._normalize_labels_csv(csv_path)
            self.logger.info("Video collection completed")
            return True
        except Exception as e:
            self.logger.error(f"Video collection failed: {e}")
            return False
        
    def _create_directories(self):
        """Create necessary directories"""
        for path in [self.data_path, self.logs_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def convert_to_ascii(self, text: str) -> str:
        """Convert Vietnamese text to ASCII-safe format"""
        text = text.lower()
        text = text.replace('đ', 'd_')  # Temporary marker
        text = text.replace('Đ', 'd_')
        
        # Unicode normalization
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore').decode('utf-8')
        
        # Handle special characters
        text = re.sub(r'[^\w\s]', '', text)
        text = text.strip()
        
        # Restore đ character
        text = text.replace('d_', 'd')
        return text
    
    def extract_keypoints(self, results) -> Optional[np.ndarray]:
        """Extract keypoints from MediaPipe results with validation"""
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
    
    def process_video(self, video_path: str, action_name: str) -> List[np.ndarray]:
        """Process single video and extract keypoints"""
        keypoints_list = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.warning(f"Could not open video: {video_path}")
                return keypoints_list
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 3rd frame for efficiency
                if frame_count % 3 == 0:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    
                    # Process with MediaPipe
                    results = self.holistic.process(image)
                    
                    # Extract keypoints
                    keypoints = self.extract_keypoints(results)
                    if keypoints is not None:
                        keypoints_list.append(keypoints)
                
                frame_count += 1
                
                # Limit frames for efficiency
                if frame_count > 300:  # Max 300 frames per video
                    break
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
        
        return keypoints_list
    
    def augment_keypoints(self, keypoints: np.ndarray) -> List[np.ndarray]:
        """Augment keypoints for data expansion"""
        augmented = [keypoints]
        
        # Add noise
        for _ in range(self.augmentation_factor - 1):
            noise = np.random.normal(0, self.noise_factor, keypoints.shape)
            augmented_keypoints = keypoints + noise
            
            # Clip to valid range
            augmented_keypoints = np.clip(augmented_keypoints, 0, 1)
            augmented.append(augmented_keypoints)
        
        return augmented
    
    def process_dataset(self) -> bool:
        """Main data processing pipeline"""
        self.logger.info("Starting data processing...")
        
        try:
            # Check dataset structure
            video_dir = Path(self.dataset_path) / 'Video'
            label_file = Path(self.dataset_path) / 'Text' / 'Label.csv'
            
            if not video_dir.exists():
                video_dir.mkdir(parents=True, exist_ok=True)
            if not label_file.exists():
                # Try to auto-collect if labels missing
                self.logger.info("Label.csv not found. Attempting to collect videos via crawler...")
                self.collect_videos_via_crawler()

            # Normalize CSV if exists in legacy format
            if label_file.exists():
                self._normalize_labels_csv(label_file)
            else:
                self.logger.error(f"Label file not found: {label_file}")
                return False

            # If no videos present in directory, try auto-collect
            if not any(video_dir.glob('*.mp4')):
                self.logger.info("Video directory is empty. Attempting to collect videos via crawler...")
                self.collect_videos_via_crawler()
            
            # Load labels
            labels_df = pd.read_csv(label_file)
            self.logger.info(f"Loaded {len(labels_df)} labels")
            
            # If none of labeled videos exist, try auto-collect then reload
            existing_matches = 0
            for _, row in labels_df.iterrows():
                vname = row.get('Video') if 'Video' in labels_df.columns else row.get('video')
                if vname and (video_dir / str(vname)).exists():
                    existing_matches += 1
            if existing_matches == 0:
                self.logger.info("No labeled videos found in folder. Attempting to collect videos via crawler...")
                self.collect_videos_via_crawler()
                # Reload labels in case CSV changed
                if label_file.exists():
                    labels_df = pd.read_csv(label_file)
                    self.logger.info(f"Reloaded labels: {len(labels_df)} rows")
            
            # Process videos
            all_keypoints = []
            all_labels = []
            action_mapping = {}
            
            for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing videos"):
                video_name = row['Video']
                action_name = row['Label']
                
                # Convert action name
                action_ascii = self.convert_to_ascii(action_name)
                
                # Create action mapping
                if action_ascii not in action_mapping:
                    action_mapping[action_ascii] = len(action_mapping)
                
                # Process video (only local files)
                video_path = video_dir / video_name
                if video_path.exists():
                    keypoints_list = self.process_video(str(video_path), action_ascii)
                    
                    if keypoints_list:
                        # Augment data
                        for keypoints in keypoints_list:
                            augmented_keypoints = self.augment_keypoints(keypoints)
                            
                            for aug_keypoints in augmented_keypoints:
                                all_keypoints.append(aug_keypoints)
                                all_labels.append(action_ascii)
                        
                        self.stats['processed_videos'] += 1
                    else:
                        self.stats['failed_videos'] += 1
                else:
                    self.logger.warning(f"Video not found: {video_path}")
                    self.stats['failed_videos'] += 1
                
                self.stats['total_videos'] += 1
            
            # Convert to numpy arrays
            if all_keypoints:
                X = np.array(all_keypoints)
                y = np.array(all_labels)
                
                # Save processed data
                np.save(Path(self.data_path) / 'keypoints.npy', X)
                np.save(Path(self.data_path) / 'labels.npy', y)
                
                # Save action mapping
                mapping_data = {
                    'created_date': datetime.now().isoformat(),
                    'total_actions': len(action_mapping),
                    'actions': action_mapping
                }
                
                mapping_path = Path(self.logs_path) / 'action_mapping.json'
                with open(mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, ensure_ascii=False, indent=2)
                
                # Save processing stats
                stats_data = {
                    'processing_date': datetime.now().isoformat(),
                    'stats': self.stats,
                    'data_shape': X.shape,
                    'unique_actions': len(np.unique(y))
                }
                
                stats_path = Path(self.logs_path) / 'processing_stats.json'
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats_data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"Data processing completed!")
                self.logger.info(f"Processed {X.shape[0]} samples with {len(np.unique(y))} actions")
                self.logger.info(f"Success rate: {self.stats['processed_videos']}/{self.stats['total_videos']}")
                
                return True
            else:
                self.logger.error("No keypoints extracted from videos")
                return False
                
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            return False
    
    def validate_data(self) -> Dict[str, Any]:
        """Validate processed data"""
        try:
            keypoints_file = Path(self.data_path) / 'keypoints.npy'
            labels_file = Path(self.data_path) / 'labels.npy'
            
            if not keypoints_file.exists() or not labels_file.exists():
                return {'valid': False, 'error': 'Data files not found'}
            
            X = np.load(keypoints_file)
            y = np.load(labels_file)
            
            validation_results = {
                'valid': True,
                'samples': X.shape[0],
                'features': X.shape[1],
                'actions': len(np.unique(y)),
                'action_distribution': dict(zip(*np.unique(y, return_counts=True))),
                'data_quality': {
                    'has_nan': np.isnan(X).any(),
                    'has_inf': np.isinf(X).any(),
                    'min_value': np.min(X),
                    'max_value': np.max(X)
                }
            }
            
            self.logger.info("Data validation completed")
            return validation_results
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def cleanup(self):
        """Cleanup resources"""
        if self.holistic:
            self.holistic.close()
        self.logger.info("Data processor cleanup completed")
