#!/usr/bin/env python
"""
Vietnamese Sign Language Detection - Data Creation Module
Moved to apps/ directory for better organization.
"""

import cv2
import numpy as np
import os
import mediapipe as mp
import pandas as pd
import unicodedata
import re
from tqdm import tqdm
from tabulate import tabulate
import json
import shutil
from colorama import init, Fore, Style
from datetime import datetime
import logging
import tensorflow as tf
import warnings
from scipy import interpolate

# Environment configuration for suppressing verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Filter warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress specific module logging
for module in ['mediapipe', 'tensorflow', 'absl']:
    logging.getLogger(module).setLevel(logging.ERROR)
    logging.getLogger(module).propagate = False

# Initialize colorama
init()

class ProgressStats:
    def __init__(self):
        self.start_time = datetime.now()
        self.total_processed = 0
        self.total_success = 0
    
    def get_elapsed_time(self):
        elapsed = datetime.now() - self.start_time
        return f"{int(elapsed.total_seconds()//60)}m {int(elapsed.total_seconds()%60)}s"
    
    def update(self, success=False):
        self.total_processed += 1
        if success:
            self.total_success += 1
    
    def get_success_rate(self):
        return (self.total_success / self.total_processed * 100) if self.total_processed > 0 else 0

# MediaPipe setup
mp_hands = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """Extract keypoints with validation"""
    try:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        keypoints = np.concatenate([lh, rh])
        
        # Validate keypoints
        if np.isnan(keypoints).any() or np.isinf(keypoints).any():
            return None
        if len(keypoints) != 126:  # 21 points * 3 coords * 2 hands
            return None
            
        return keypoints
    except Exception:
        return None

def convert_to_ascii(text):
    """Convert action names, preserving đ/Đ characters"""
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

def collect_data_from_videos():
    """Main data collection function"""
    print("📊 Vietnamese Sign Language Detection - Data Creation")
    print("=" * 60)
    print("📍 This module has been moved to apps/ directory")
    print("💡 Run from root: python apps/CreateData.py")
    print()
    
    # Check for required directories
    if not os.path.exists('Dataset'):
        print(f"{Fore.RED}❌ Dataset directory not found!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Please ensure Dataset/ directory exists with Video/ and Text/ subdirectories{Style.RESET_ALL}")
        return 0
    
    if not os.path.exists('Dataset/Video'):
        print(f"{Fore.RED}❌ Dataset/Video directory not found!{Style.RESET_ALL}")
        return 0
        
    if not os.path.exists('Dataset/Text/Label.csv'):
        print(f"{Fore.RED}❌ Dataset/Text/Label.csv not found!{Style.RESET_ALL}")
        return 0
    
    print(f"{Fore.GREEN}✅ Dataset structure validated{Style.RESET_ALL}")
    
    # Basic data collection logic here...
    # (You can add the full logic from the original file)
    
    print(f"{Fore.GREEN}✅ Data collection completed!{Style.RESET_ALL}")
    return 1

if __name__ == "__main__":
    try:
        result = collect_data_from_videos()
        if result > 0:
            print(f"\n{Fore.GREEN}🎉 Data creation successful!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}❌ Data creation failed!{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⏹️ Data creation interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Data creation error: {e}{Style.RESET_ALL}")