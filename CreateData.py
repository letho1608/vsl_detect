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

# Ẩn các cảnh báo không cần thiết
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('mediapipe.python._framework_bindings.landmark_pb2').setLevel(logging.ERROR)
logging.getLogger('mediapipe.python._framework_bindings.packet_creator').setLevel(logging.ERROR)
logging.getLogger('mediapipe.python._framework_bindings.packet_getter').setLevel(logging.ERROR)
logging.getLogger('mediapipe.python._framework_bindings.timestamp').setLevel(logging.ERROR)
logging.getLogger('mediapipe.python._framework_bindings.validated_graph_config').setLevel(logging.ERROR)

# Suppress specific MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Add this line

# Disable tensorflow logging
tf.get_logger().setLevel('ERROR')

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
    """Extract keypoints và kiểm tra tính hợp lệ"""
    try:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        keypoints = np.concatenate([lh, rh])
        
        # Kiểm tra tính hợp lệ của keypoints
        if np.isnan(keypoints).any() or np.isinf(keypoints).any():
            return None
        if len(keypoints) != 126:  # 21 điểm * 3 tọa độ * 2 bàn tay
            return None
            
        return keypoints
    except Exception:
        return None

def convert_to_ascii(text):
    """Chuyển đổi tên hành động, giữ nguyên ký tự đ/Đ"""
    # Xử lý riêng ký tự đ/Đ
    text = text.lower()
    text = text.replace('đ', 'd_')  # Đánh dấu tạm thời
    text = text.replace('Đ', 'd_')
    
    # Chuẩn hóa unicode
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    # Xử lý ký tự đặc biệt
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    
    # Khôi phục lại ký tự đ
    text = text.replace('d_', 'd')
    return text

def create_action_folder(data_path, action_ascii):
    """Tạo thư mục cho một hành động với cấu trúc mới"""
    action_path = os.path.join(data_path, action_ascii)
    os.makedirs(action_path, exist_ok=True)
    return action_path

def save_action_mapping(selected_actions, log_path='Logs'):
    """Lưu mapping của các hành động đã chọn vào thư mục Logs"""
    os.makedirs(log_path, exist_ok=True)
    mapping_file = os.path.join(log_path, 'action_mapping.json')
    
    mapping = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'actions': {
            convert_to_ascii(action): action 
            for action in selected_actions
        }
    }
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"\n{Fore.CYAN}[INFO] Đã lưu mapping vào: {Fore.GREEN}{mapping_file}{Style.RESET_ALL}")

def load_action_mapping(log_path='Logs'):
    """Đọc mapping từ thư mục Logs"""
    mapping_file = os.path.join(log_path, 'action_mapping.json')
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            return mapping_data['actions']
    except FileNotFoundError:
        print(f"{Fore.RED}[ERROR] Không tìm thấy file mapping: {mapping_file}{Style.RESET_ALL}")
        return {}
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Lỗi khi đọc file mapping: {str(e)}{Style.RESET_ALL}")
        return {}

def get_action_name(action_ascii, mapping=None):
    """Chuyển đổi tên action ASCII sang tên đầy đủ"""
    if mapping is None:
        mapping = load_action_mapping()
    return mapping.get(action_ascii, action_ascii)

def save_progress_state(state_data, log_path='Logs'):
    """Lưu trạng thái hiện tại vào thư mục Logs"""
    os.makedirs(log_path, exist_ok=True)
    state_file = os.path.join(log_path, 'progress_state.json')
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state_data, f, ensure_ascii=False, indent=2)

def load_progress_state(log_path='Logs'):
    """Đọc trạng thái từ thư mục Logs"""
    state_file = os.path.join(log_path, 'progress_state.json')
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def interpolate_keypoints(keypoints_sequence, target_length=60):
    """Nội suy chuỗi keypoints để đạt độ dài mong muốn"""
    if len(keypoints_sequence) == 0:
        return None
        
    # Tạo trục thời gian cho dữ liệu gốc và target
    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_length)
    
    # Nội suy cho từng feature
    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_length, num_features))
    
    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]
        
        # Sử dụng cubic spline interpolation
        interpolator = interpolate.interp1d(
            original_times, feature_values, 
            kind='cubic',
            bounds_error=False,
            fill_value="extrapolate"
        )
        interpolated_sequence[:, feature_idx] = interpolator(target_times)
    
    return interpolated_sequence

def process_video_sequence(video_path, holistic, sequence_length=60):
    """Xử lý video và trích xuất keypoints với nội suy"""
    sequence_frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Xác định step size để lấy mẫu frames
    step = max(1, total_frames // 100)  # Lấy tối đa 100 frames để xử lý
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Chỉ xử lý các frame cách đều nhau
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue
            
        try:
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            
            if keypoints is not None:
                sequence_frames.append(keypoints)
                
        except Exception as e:
            continue
            
    cap.release()
    
    # Kiểm tra số lượng frames thu được
    if len(sequence_frames) < 3:  # Cần ít nhất 3 frames cho nội suy cubic
        return None
        
    # Nội suy để có đúng sequence_length frames
    try:
        interpolated_sequence = interpolate_keypoints(sequence_frames, sequence_length)
        return interpolated_sequence
    except Exception as e:
        print(f"Lỗi khi nội suy: {str(e)}")
        return None

def collect_data_from_videos():
    """Thu thập dữ liệu từ video với khả năng tiếp tục"""
    DATA_PATH = os.path.join('Data')
    DATASET_PATH = os.path.join('Dataset')
    LOG_PATH = os.path.join('Logs')
    
    os.makedirs(LOG_PATH, exist_ok=True)
    
    # Load data first
    label_file = os.path.join(DATASET_PATH, 'Text', 'Label.csv')
    video_folder = os.path.join(DATASET_PATH, 'Video')
    df = pd.read_csv(label_file)
    
    # Kiểm tra trạng thái từ lần chạy trước
    previous_state = load_progress_state(LOG_PATH)
    if previous_state and os.path.exists(DATA_PATH):
        while True:
            choice = input(f"\n{Fore.YELLOW}Phát hiện dữ liệu từ lần chạy trước. Bạn có muốn tiếp tục? (y/n): {Style.RESET_ALL}").lower()
            if choice in ['y', 'n']:
                break
            print(f"{Fore.RED}Vui lòng nhập 'y' hoặc 'n'{Style.RESET_ALL}")
        
        if choice == 'y':
            selected_actions = previous_state['selected_actions']
            num_actions = len(selected_actions)
            print(f"{Fore.GREEN}Tiếp tục với {num_actions} hành động từ lần trước{Style.RESET_ALL}")
            df_filtered = df[df['TEXT'].isin(selected_actions)]
        else:
            if os.path.exists(DATA_PATH):
                shutil.rmtree(DATA_PATH)
            previous_state = None
    
    if not previous_state:
        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(LOG_PATH, exist_ok=True)
        
        no_sequences = 60
        sequence_length = 60
        
        # Thêm input số lượng hành động
        total_actions = len(df['TEXT'].unique())
        while True:
            try:
                num_actions = int(input(f"\n{Fore.CYAN}Nhập số lượng hành động cần thu thập (tối đa {total_actions}): {Style.RESET_ALL}"))
                if 1 <= num_actions <= total_actions:
                    break
                print(f"{Fore.RED}Vui lòng nhập số từ 1 đến {total_actions}{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Vui lòng nhập một số nguyên hợp lệ{Style.RESET_ALL}")

        # Chọn ngẫu nhiên số lượng hành động theo yêu cầu
        selected_actions = np.random.choice(df['TEXT'].unique(), num_actions, replace=False)
        df_filtered = df[df['TEXT'].isin(selected_actions)]
        
        # Lưu mapping trong thư mục Logs
        save_action_mapping(selected_actions, LOG_PATH)
        
        print(f"\n{Fore.CYAN}[INFO] Đã chọn {Fore.GREEN}{num_actions}{Fore.CYAN} hành động từ tổng số {Fore.GREEN}-{total_actions}{Fore.CYAN} hành động{Style.RESET_ALL}")
    
    stats = ProgressStats()
    print(f"\n{Fore.CYAN}[{datetime.now().strftime('%H:%M:%S')}] Bắt đầu xử lý dữ liệu{Style.RESET_ALL}")

    # Trong vòng lặp xử lý action
    with mp_hands.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action_idx, (action, group) in enumerate(tqdm(df_filtered.groupby('TEXT'), desc="Xử lý các hành động"), 1):
            action_ascii = convert_to_ascii(action)
            action_path = create_action_folder(DATA_PATH, action_ascii)
            
            # Xác định sequence bắt đầu
            start_sequence = 0
            if previous_state and action in previous_state['progress']:
                start_sequence = previous_state['progress'][action]
            
            for sequence in range(start_sequence, no_sequences):
                sequence_folder = os.path.join(action_path, str(sequence))
                os.makedirs(sequence_folder, exist_ok=True)
                
                video_row = group.sample(n=1).iloc[0]
                video_path = os.path.join(video_folder, video_row['VIDEO'])
                
                if not os.path.exists(video_path):
                    print(f"\nLỗi: Không tìm thấy {video_path}")
                    continue
                
                # Xử lý video và nội suy
                interpolated_sequence = process_video_sequence(
                    video_path, 
                    holistic, 
                    sequence_length
                )
                
                if interpolated_sequence is not None:
                    # Lưu các frames đã nội suy
                    for frame_idx, keypoints in enumerate(interpolated_sequence):
                        npy_path = os.path.join(sequence_folder, f"{frame_idx}.npy")
                        np.save(npy_path, keypoints)
                    stats.update(success=True)
                else:
                    stats.update(success=False)
                    continue
                
                # Lưu trạng thái sau mỗi sequence
                current_state = {
                    'selected_actions': selected_actions.tolist(),
                    'progress': {
                        action: sequence + 1
                    }
                }
                if previous_state and 'progress' in previous_state:
                    current_state['progress'].update(previous_state['progress'])
                save_progress_state(current_state, LOG_PATH)
                
                # Update progress display
                success_rate = stats.get_success_rate()
                status_color = Fore.GREEN if success_rate >= 80 else Fore.YELLOW if success_rate >= 50 else Fore.RED
                print(f"\r{Fore.CYAN}Xử lý hành động ({action_idx}/{len(df['TEXT'].unique())}): {Fore.GREEN}{action}{Style.RESET_ALL} | "
                      f"{Fore.CYAN}Sequence: {sequence+1}/{no_sequences} | "
                      f"{Fore.CYAN}Thành công: {status_color}{stats.total_success}{Style.RESET_ALL} | "
                      f"{Fore.CYAN}Tỷ lệ: {status_color}{success_rate:.1f}%{Style.RESET_ALL}", end='')

    print(f"\n\n{Fore.YELLOW}{'='*50}")
    print(f"{Fore.CYAN}Kết quả thu thập dữ liệu:{Style.RESET_ALL}")
    total_sequences = stats.total_success
    total_videos = len(df)
    print(f"{Fore.GREEN}➤ Tổng số sequence đã lưu: {total_sequences}")
    print(f"➤ Tỷ lệ thành công: {total_sequences/total_videos:.1%}")
    print(f"➤ Tổng số hành động: {len(df['TEXT'].unique())}{Style.RESET_ALL}")
    
    # Create logging data
    overall_progress = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_sequences': total_sequences,
        'total_videos': total_videos,
        'success_rate': float(total_sequences/total_videos),
        'total_actions': len(df['TEXT'].unique()),
        'processed_actions': num_actions,
        'elapsed_time': stats.get_elapsed_time(),
        'videos_processed': stats.total_processed,
        'sequences_success': stats.total_success,
        'success_rate_detailed': float(stats.get_success_rate())
    }
    
    log_file_path = os.path.join(LOG_PATH, 'data_collection_log.json')
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        json.dump(overall_progress, log_file, ensure_ascii=False, indent=2)
    
    print(f"\n{Fore.YELLOW}{'='*50}")
    print(f"{Fore.CYAN}Tổng kết:{Style.RESET_ALL}")
    print(f"├─ Tổng thời gian: {stats.get_elapsed_time()}")
    print(f"├─ Videos đã xử lý: {Fore.GREEN}{stats.total_processed}{Style.RESET_ALL}")
    print(f"├─ Sequences thành công: {Fore.GREEN}{stats.total_success}{Style.RESET_ALL}")
    print(f"└─ Tỷ lệ thành công: {Fore.GREEN}{stats.get_success_rate():.1f}%{Style.RESET_ALL}")
    
    return stats.total_success

def count_collected_data():
    """Đếm số lượng hành động đã thu thập"""
    count = len(next(os.walk('Data'))[1]) if os.path.exists('Data') else 0  # Changed from 'Data_test' to 'Data'
    print(f"\n{Fore.CYAN}[INFO] Số lượng hành động đã thu thập: {Fore.GREEN}{count}{Style.RESET_ALL}")
    return count

if __name__ == "__main__":
    print(f"\n{Fore.YELLOW}{'='*50}")
    print(f"{Fore.CYAN}Bắt đầu thu thập dữ liệu...{Style.RESET_ALL}")
    if os.path.exists(os.path.join('Logs', 'progress_state.json')):
        print(f"{Fore.YELLOW}Tìm thấy trạng thái từ lần chạy trước{Style.RESET_ALL}")
    total_sequences = collect_data_from_videos()
    count_collected_data()