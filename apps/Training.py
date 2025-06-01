#!/usr/bin/env python
import os
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import sys
import logging
from multiprocessing import Pool, cpu_count  # Add this import
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tabulate import tabulate
from colorama import Fore, Style, init
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


init(autoreset=True)

# Buộc sử dụng CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Thêm các biến môi trường trước khi tạo mô hình
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt cảnh báo tối ưu hóa oneDNN

# Kiểm tra và cấu hình GPU/CPU
def setup_gpu():
    """Check and configure GPU if available using TensorFlow"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\n{Fore.GREEN}Đã tìm thấy GPU:{Style.RESET_ALL}")
            for gpu in gpus:
                print(f"{Fore.CYAN}- {gpu.name}")
            return True
        except RuntimeError as e:
            print(f"\n{Fore.YELLOW}Lỗi khi cấu hình GPU: {e}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Không tìm thấy GPU, sử dụng CPU...{Style.RESET_ALL}")
    return False

# Đường dẫn đầu vào và đầu ra
DATA_PATH = 'Data'  # Thư mục chứa dữ liệu đã xử lý
MODEL_PATH = 'Models'  # Thư mục chứa model đã train
LOG_PATH = 'Logs'  # Thư mục chứa logs

def setup_logging():
    """Simplified logging setup"""
    os.makedirs(LOG_PATH, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOG_PATH, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("=== TRAINING STARTED ===")
    logging.info(f"Log file: {log_file}")

def load_action_mapping():
    """Load mapping từ file json"""
    mapping_file = os.path.join('Logs', 'action_mapping.json')  # Changed from 'Models' to 'Logs'
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            print(f"\n{Fore.GREEN}Đã tải mapping:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- Ngày tạo: {mapping_data['created_date']}")
            print(f"{Fore.CYAN}- Tổng số hành động: {mapping_data['total_actions']}")
            return mapping_data['actions']
    except Exception as e:
        print(f"{Fore.RED}Lỗi khi tải mapping: {str(e)}{Style.RESET_ALL}")
        return None

# ... (rest of the training code continues)

if __name__ == "__main__":
    print("🧠 Vietnamese Sign Language Detection - Training Module")
    print("=" * 60)
    print("📍 This file has been moved to apps/ directory")
    print("💡 Run from root: python apps/Training.py")
    
    # Continue with original training logic
    setup_logging()
    setup_gpu()
    
    try:
        history, model = train_model()
        if history and model:
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed!")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training error: {e}")