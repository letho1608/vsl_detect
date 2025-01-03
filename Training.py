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

def save_action_mapping(selected_actions, log_path='Logs'):
    """Lưu mapping của các hành động đã chọn"""
    os.makedirs(log_path, exist_ok=True)
    mapping_file = os.path.join(log_path, 'action_mapping.json')
    
    mapping = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'created_date': datetime.now().strftime('%Y-%m-%d'),  # Thêm trường này
        'total_actions': len(selected_actions),
        'actions': {
            action: {
                'processed_name': convert_to_ascii(action),
                'original_name': action
            }
            for action in selected_actions
        }
    }
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

def get_actions_from_directory():
    """Lấy danh sách hành động với xử lý lỗi tốt hơn"""
    # Kiểm tra và tạo thư mục Data nếu chưa tồn tại
    if not os.path.exists(DATA_PATH):
        try:
            os.makedirs(DATA_PATH)
            print(f"{Fore.YELLOW}Đã tạo thư mục Data{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Không thể tạo thư mục Data: {str(e)}{Style.RESET_ALL}")
            return None

    # Thử tải mapping trước
    try:
        mapping_file = os.path.join('Logs', 'action_mapping.json')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                if 'actions' in mapping_data:
                    actions = sorted([info['processed_name'] 
                                   for info in mapping_data['actions'].values()])
                    print(f"\n{Fore.GREEN}Đã tải {len(actions)} hành động từ mapping{Style.RESET_ALL}")
                    return np.array(actions)
    except Exception as e:
        print(f"{Fore.YELLOW}Lỗi khi đọc mapping: {str(e)}{Style.RESET_ALL}")

    # Quét thư mục Data nếu không có mapping
    try:
        actions = []
        for item in os.listdir(DATA_PATH):
            if os.path.isdir(os.path.join(DATA_PATH, item)):
                actions.append(item)
        
        if not actions:
            print(f"{Fore.YELLOW}Không tìm thấy thư mục con trong Data/{Style.RESET_ALL}")
            return None
            
        print(f"\n{Fore.GREEN}Đã tìm thấy {len(actions)} hành động từ thư mục:{Style.RESET_ALL}")
        for idx, action in enumerate(sorted(actions), 1):
            print(f"{Fore.CYAN}{idx}. {action}")
            
        return np.array(sorted(actions))
        
    except Exception as e:
        print(f"{Fore.RED}Lỗi khi quét thư mục Data: {str(e)}{Style.RESET_ALL}")
        return None

actions = get_actions_from_directory()
if actions is None:
    print("Không thể tải danh sách nhãn. Chương trình sẽ kết thúc.")
    sys.exit(1)

no_sequences = 60
sequence_length = 60

def build_model():
    """Xây dựng mô hình LSTM Keras"""
    # Tạo lớp đầu vào rõ ràng
    inputs = tf.keras.Input(shape=(60, 126))
    
    # Khối LSTM thứ nhất
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(inputs)
    x = BatchNormalization()(x)
    
    # Khối LSTM thứ hai
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = BatchNormalization()(x)
    
    # Khối LSTM thứ ba
    x = Bidirectional(LSTM(256, dropout=0.3))(x)
    x = BatchNormalization()(x)
    
    # Các lớp Dense
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    # Lớp đầu ra
    outputs = Dense(len(actions), activation='softmax')(x)
    
    # Tạo mô hình
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Biên dịch mô hình
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_latest_checkpoint():
    """Load latest Keras checkpoint with full training state"""
    checkpoint_path = os.path.join('Models', 'checkpoints', 'checkpoint_model.keras')
    metadata_path = os.path.join('Models', 'checkpoints', 'training_metadata.json')
    
    if os.path.exists(checkpoint_path) and os.path.exists(metadata_path):
        try:
            # Load Keras model
            keras_model = tf.keras.models.load_model(checkpoint_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Convert Keras model to PyTorch format
            model = build_model()
            # ... (weights transfer logic)
            
            training_state = {
                'epoch': metadata['epoch'],
                'best_accuracy': metadata['best_accuracy'],
                'history': metadata.get('history', None),
                'prev_val_loss': metadata.get('prev_val_loss', float('inf')),
                'table_data': metadata.get('table_data', [])
            }
            
            print(f"\nĐã tìm thấy checkpoint:")
            print(f"- Epoch: {training_state['epoch'] + 1}")
            print(f"- Độ chính xác tốt nhất: {training_state['best_accuracy']:.4%}")
            
            return model, training_state
            
        except Exception as e:
            print(f"Lỗi khi tải checkpoint: {str(e)}")
            return None, None
    return None, None

def save_checkpoint(model, epoch, best_accuracy, history, table_data, is_best=False):
    """Save Keras checkpoint"""
    os.makedirs(os.path.join(MODEL_PATH, 'checkpoints'), exist_ok=True)
    
    # Save model
    checkpoint_path = os.path.join(MODEL_PATH, 'checkpoints', f'checkpoint_model_{epoch:03d}.keras')
    model.save(checkpoint_path)
    
    # Save metadata
    metadata = {
        'epoch': epoch,
        'best_accuracy': float(best_accuracy),
        'history': history,
        'table_data': table_data
    }
    
    metadata_path = os.path.join(MODEL_PATH, 'checkpoints', 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
        
    if is_best:
        best_path = os.path.join(MODEL_PATH, 'best_model.keras')
        model.save(best_path)

def load_sequence_chunk(args):
    """Load a chunk of sequences in parallel với kiểm tra chi tiết"""
    action, sequence_range = args
    sequences = []
    labels = []
    
    try:
        # Kiểm tra thư mục action có tồn tại không
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            logging.error(f"Thư mục hành động không tồn tại: {action_path}")
            return sequences, labels

        # In thông tin debug
        logging.info(f"\nĐang xử lý hành động: {action}")
        logging.info(f"Đường dẫn: {action_path}")
        logging.info(f"Dải sequence cần xử lý: {list(sequence_range)}")
        
        # Liệt kê tất cả sequence có sẵn
        available_sequences = [s for s in os.listdir(action_path) 
                             if os.path.isdir(os.path.join(action_path, s))]
        logging.info(f"Các sequence có sẵn: {available_sequences}")

        for sequence_idx in sequence_range:
            sequence_path = os.path.join(action_path, str(sequence_idx))
            
            # Kiểm tra chi tiết sequence
            if not os.path.exists(sequence_path):
                logging.warning(f"Bỏ qua sequence {sequence_idx} - Không tồn tại đường dẫn: {sequence_path}")
                continue

            # Kiểm tra và load frames
            frame_files = sorted([f for f in os.listdir(sequence_path) 
                                if f.endswith('.npy')],
                               key=lambda x: int(x.split('.')[0]))
            
            logging.info(f"Sequence {sequence_idx}: Tìm thấy {len(frame_files)} frames")
            
            if len(frame_files) != sequence_length:
                logging.error(f"Bỏ qua sequence {sequence_idx}: Không đủ frames ({len(frame_files)}/{sequence_length})")
                continue

            # Load và kiểm tra từng frame
            window = []
            valid_sequence = True
            
            for frame_file in frame_files:
                file_path = os.path.join(sequence_path, frame_file)
                try:
                    frame_data = np.load(file_path, allow_pickle=True)
                    if frame_data.shape != (126,):
                        logging.error(f"Frame không đúng kích thước: {file_path} ({frame_data.shape})")
                        valid_sequence = False
                        break
                    window.append(frame_data)
                except Exception as e:
                    logging.error(f"Lỗi load frame {file_path}: {str(e)}")
                    valid_sequence = False
                    break

            if valid_sequence and len(window) == sequence_length:
                sequences.append(window)
                labels.append(actions.tolist().index(action))
                logging.info(f"Đã load thành công sequence {sequence_idx}")
            else:
                logging.warning(f"Bỏ qua sequence {sequence_idx} do không hợp lệ")

    except Exception as e:
        logging.error(f"Lỗi xử lý hành động {action}: {str(e)}", exc_info=True)
    
    logging.info(f"Kết thúc xử lý {action}: {len(sequences)} sequences hợp lệ")
    return sequences, labels

def prepare_training_data():
    """Optimized data preparation using multiprocessing"""
    logging.info("Bắt đầu chuẩn bị dữ liệu huấn luyện")
    
    try:
        # Tạo header cho bảng
        headers = ['STT', 'Hành động', 'Tiến độ', 'Số mẫu', 'Trạng thái']
        table_data = []
        all_sequences = []
        all_labels = []
        
        print("Đang quét dữ liệu training...")
        total_sequences = 0
        total_frames = 0
        
        # Kiểm tra số lượng dữ liệu
        for action in actions:
            action_path = os.path.join(DATA_PATH, action)
            if not os.path.exists(action_path):
                logging.error(f"Thư mục không tồn tại: {action_path}")
                continue
                
            sequences = [s for s in os.listdir(action_path) 
                        if os.path.isdir(os.path.join(action_path, s))]
            total_sequences += len(sequences)
            
            for seq in sequences:
                seq_path = os.path.join(action_path, seq)
                frames = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
                total_frames += len(frames)
        
        print(f"Tổng số sequences: {total_sequences}")
        print(f"Tổng số frames: {total_frames}")
        
        if total_sequences == 0:
            logging.error("Không tìm thấy dữ liệu training!")
            return None, None, None, None  # Trả về 4 giá trị None thay vì 2
        
        # Chia nhỏ công việc cho multiprocessing
        chunk_size = 10  # Số sequence xử lý mỗi lần
        num_processes = max(1, cpu_count() - 1)  # Giữ lại 1 core cho hệ thống
        
        print(f"Sử dụng {num_processes} processes để xử lý dữ liệu")
        
        with Pool(processes=num_processes) as pool:
            for action_idx, action in enumerate(actions):
                # Chia sequences thành các chunks
                sequence_chunks = [
                    (action, range(i, min(i + chunk_size, no_sequences)))
                    for i in range(0, no_sequences, chunk_size)
                ]
                
                # Xử lý song song các chunks
                results = list(tqdm(
                    pool.imap(load_sequence_chunk, sequence_chunks),
                    total=len(sequence_chunks),
                    desc=f"Xử lý '{action}'",
                    leave=False
                ))
                
                # Gom kết quả
                action_sequences = []
                action_labels = []
                for seq, lab in results:
                    action_sequences.extend(seq)
                    action_labels.extend(lab)
                
                # Cập nhật bảng trạng thái
                progress = f"{len(action_sequences)}/{no_sequences}"
                status = "✓" if len(action_sequences) == no_sequences else "⚠"
                table_data.append([
                    f"{action_idx + 1}/{len(actions)}",
                    action,
                    progress,
                    len(action_sequences),
                    status
                ])
                
                # Thêm vào tổng hợp
                all_sequences.extend(action_sequences)
                all_labels.extend(action_labels)
                
                # Hiển thị tiến độ
                os.system('cls' if os.name == 'nt' else 'clear')
                print("\nTiến độ chuẩn bị dữ liệu:")
                print(tabulate(table_data, headers=headers, tablefmt='grid'))
                print(f"\nTổng số mẫu hiện tại: {len(all_sequences)}")
    
        # Kiểm tra dữ liệu trước khi trả về
        if len(all_sequences) == 0:
            logging.error("Không có dữ liệu sau khi xử lý!")
            return None, None, None, None
            
        # Chuyển đổi dữ liệu
        X = np.array(all_sequences)
        y = to_categorical(all_labels).astype(int)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    except Exception as e:
        logging.error(f"Lỗi khi chuẩn bị dữ liệu: {str(e)}")
        return None, None, None, None  # Trả về 4 giá trị None thay vì 2

def train_model(epochs=50, batch_size=32, initial_epoch=0):
    """Train model using Keras with better monitoring"""
    # Setup model
    model = build_model()
    model.best_accuracy = 0
    
    # Setup callbacks
    table_data = []
    headers = ['Epoch', 'Loss', 'Accuracy', 'Val Loss', 'Val Acc', 'LR', 'Status']
    
    callbacks = [
        ModelCheckpoint(
            os.path.join('Models', 'checkpoints', 'model_{epoch:02d}.keras'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        ),
        TrainingCallback(table_data, headers)
    ]
    
    # Prepare data với kiểm tra lỗi
    try:
        X_train, X_test, y_train, y_test = prepare_training_data()
        if X_train is None or X_test is None or y_train is None or y_test is None:
            logging.error("Không thể chuẩn bị dữ liệu training")
            return None, None
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=0  # Tắt output mặc định
        )
        
        # Save final model
        save_model(model)
        
        return history.history, model
        
    except KeyboardInterrupt:
        print("\nHuấn luyện bị dừng bởi người dùng")
        return history.history, model
    except Exception as e:
        logging.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
        return None, None

# Thêm class TensorBoard callback
class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, table_data, headers):
        super().__init__()
        self.table_data = table_data
        self.headers = headers
        
    def on_epoch_end(self, epoch, logs=None):
        # Cập nhật bảng trạng thái
        status = ''
        if logs.get('val_accuracy', 0) > self.model.best_accuracy:
            status += f"{Fore.RED}🔥{Style.RESET_ALL}"
            self.model.best_accuracy = logs['val_accuracy']
        elif logs.get('val_accuracy', 0) >= self.model.best_accuracy * 0.95:
            status += f"{Fore.GREEN}✅{Style.RESET_ALL}"
        else:
            status += f"{Fore.YELLOW}⚠️{Style.RESET_ALL}"
        
        # Lấy learning rate hiện tại
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            
        self.table_data.append([
            f'{epoch+1}/{self.params["epochs"]}',
            f'{logs["loss"]:.4f}',
            f'{logs["accuracy"]:.4%}',
            f'{logs["val_loss"]:.4f}',
            f'{logs["val_accuracy"]:.4%}',
            f'{current_lr:.6f}',  # Sử dụng learning rate đã lấy
            status
        ])
        
        # Hiển thị bảng
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"\n{Fore.GREEN}Kết quả huấn luyện:{Style.RESET_ALL}")
        print(tabulate(self.table_data, self.headers, tablefmt='grid'))

def save_model(model, path='Models/final_model.keras'):
    """Lưu mô hình Keras"""
    model.save(path)
    print(f"Đã lưu mô hình vào {path}")

def load_model(path='Models/best_model.keras'):
    """Tải mô hình Keras"""
    return tf.keras.models.load_model(path)

def plot_training_history(history):
    """Plot training metrics from dictionary history"""
    try:
        # Kiểm tra dữ liệu trước khi vẽ
        if not all(key in history for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']):
            print("Thiếu dữ liệu history!")
            return
            
        if len(history['loss']) == 0:
            print("Không có dữ liệu training để vẽ!")
            return

        plt.figure(figsize=(12, 4))
        
        # Plot với nhiều thông tin hơn
        epochs = range(1, len(history['loss']) + 1)
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        train_acc = plt.plot(epochs, history['accuracy'], 'b-', label='Độ chính xác huấn luyện')
        val_acc = plt.plot(epochs, history['val_accuracy'], 'r-', label='Độ chính xác kiểm thử')
        plt.title('Độ chính xác của mô hình')
        plt.xlabel('Epoch')
        plt.ylabel('Độ chính xác')
        plt.grid(True)
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        train_loss = plt.plot(epochs, history['loss'], 'b-', label='Mất mát huấn luyện')
        val_loss = plt.plot(epochs, history['val_loss'], 'r-', label='Mất mát kiểm thử')
        plt.title('Độ mất mát của mô hình')
        plt.xlabel('Epoch')
        plt.ylabel('Mất mát')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join('Logs', 'training_history.png'))
        plt.close()
        
        # In thông tin để debug
        print("\nThống kê training:")
        print(f"Epochs đã train: {len(epochs)}")
        print(f"Độ chính xác cuối: {history['accuracy'][-1]:.4f}")
        print(f"Loss cuối: {history['loss'][-1]:.4f}")
        print("Đã lưu biểu đồ huấn luyện vào thư mục Logs")
        
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ: {str(e)}")
        # Print thêm thông tin debug
        print("History dict:", history.keys())
        print("Lengths:", {k: len(v) for k, v in history.items()})

def evaluate_model(model, X_test, y_test):
    """Evaluate model với tên hành động gốc"""
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    acc = accuracy_score(y_test_classes, y_pred_classes)
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Print results
    print("\nKết quả đánh giá mô hình:")
    print(f"Độ chính xác: {acc:.4f}")
    print("\nMa trận nhầm lẫn:")
    print(conf_matrix)
    
    # Thêm tên gốc vào kết quả nếu có mapping
    mapping = load_action_mapping()
    if mapping:
        # Tạo mapping ngược từ tên đã xử lý về tên gốc
        reverse_mapping = {
            v['processed_name']: k 
            for k, v in mapping.items()
        }
        
        # In kết quả với cả tên gốc
        print("\nKết quả theo tên gốc:")
        for i, action in enumerate(actions):
            original = reverse_mapping.get(action, action)
            print(f"{i+1}. {action} ({original})")
            # In các metric khác...
    
    # Save model metrics
    with open(os.path.join('Logs', 'model_evaluation.txt'), 'w') as f:
        f.write(f"Kết quả đánh giá mô hình:\n")
        f.write(f"Độ chính xác: {acc:.4f}\n")
        f.write(f"\nMa trận nhầm lẫn:\n")
        f.write(str(conf_matrix))

if __name__ == "__main__":
    # Khởi tạo logging
    setup_logging()
    
    try:
        logging.info("=== KHỞI TẠO HUẤN LUYỆN ===")
        print(f"{Fore.GREEN}=== HỆ THỐNG NHẬN DẠNG NGÔN NGỮ KÝ HIỆU ==={Style.RESET_ALL}")
        
        # Kiểm tra GPU
        has_gpu = setup_gpu()
        
        # Load actions với mapping
        actions = get_actions_from_directory()
        if actions is None:
            print("Không thể tải danh sách hành động. Chương trình sẽ kết thúc.")
            sys.exit(1)
        
        # Thử tải checkpoint với giá trị trả về đã sửa
        loaded_model, loaded_state = load_latest_checkpoint()
        
        if loaded_model is not None and loaded_state is not None:
            print(f"Tiếp tục huấn luyện từ epoch {loaded_state['epoch'] + 1}")
            model = loaded_model
            initial_epoch = loaded_state['epoch'] + 1
        else:
            print("\nKhởi tạo mô hình mới...")
            model = build_model()
            initial_epoch = 0
        
        epochs = 50
        batch_size = 8 if has_gpu else 32
        
        train_history, trained_model = train_model(epochs=epochs, batch_size=batch_size, initial_epoch=initial_epoch)
        
        if trained_model is not None and train_history is not None:
            # Plot training history
            plot_training_history(train_history)
            print(f"{Fore.GREEN}Đã lưu mô hình và biểu đồ huấn luyện thành công!{Style.RESET_ALL}")
            
    except Exception as e:
        logging.error(f"Lỗi không xử lý được: {str(e)}", exc_info=True)
        print(f"{Fore.RED}Lỗi không xử lý được: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

