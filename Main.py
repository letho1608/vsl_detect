import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from collections import deque
from colorama import Fore, Style, init
import gtts
import threading
import tempfile
import json
from playsound import playsound, PlaysoundException
import time  # Add this import at the top
from pygame import mixer  # Add this import at top
import os.path
import win32api  # Add this import at top

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ Thống Nhận Dạng Ngôn Ngữ Ký Hiệu")
        self.setGeometry(100, 100, 1200, 800)
        
        # Khởi tạo các biến
        self.model = None
        self.actions = None
        self.sequence = deque(maxlen=60)
        self.predictions = deque(maxlen=5)
        self.cap = None
        self.timer = None
        self.show_landmarks = True
        self.is_recording = False
        self.last_prediction = None  # Thêm biến để theo dõi dự đoán cuối cùng
        self.consecutive_count = 0   # Đếm số lần dự đoán liên tiếp giống nhau
        self.min_consecutive = 3     # Số lần tối thiểu để chấp nhận một dự đoán
        self.is_camera_on = False    # Thêm biến theo dõi trạng thái
        
        # Thêm các biến mới
        self.is_speaking = False
        self.temp_audio_file = None
        self.current_sentence = []
        self.is_auto_speak = False
        self.auto_speak_threshold = 5  # Số từ tối thiểu để tự động đọc
        self.action_mapping = None  # Thêm biến để lưu mapping
        self.last_spoken_text = ""  # Thêm biến để theo dõi text đã đọc
        
        # Thay đổi cách tạo thư mục Voice
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.voice_dir = os.path.join(current_dir, 'Voice')
            if not os.path.exists(self.voice_dir):
                os.makedirs(self.voice_dir)
            print(f"Voice directory: {self.voice_dir}")  # Debug log
        except Exception as e:
            print(f"Error creating voice directory: {e}")
            # Fallback to current directory
            self.voice_dir = os.path.join(os.getcwd(), 'Voice')
            os.makedirs(self.voice_dir, exist_ok=True)
        
        # Khởi tạo MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Add image dimensions for landmark projection
        self.image_width = 640  # Default camera width
        self.image_height = 480 # Default camera height
        
        self.setup_ui()
        self.load_model_and_actions()
        
        # Initialize pygame mixer
        mixer.init()
        
    def setup_ui(self):
        # Widget chính với QSplitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Thanh công cụ
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Thêm actions cho toolbar
        self.setup_toolbar_actions(toolbar)
        
        # Splitter chính
        splitter = QSplitter(Qt.Horizontal)
        
        # Panel bên trái (Camera)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Camera view với border và shadow
        camera_frame = QFrame()
        camera_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        camera_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #cccccc;
                border-radius: 10px;
                background-color: white;
            }
        """)
        camera_layout = QVBoxLayout(camera_frame)
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(800, 600)
        camera_layout.addWidget(self.camera_label)
        
        left_layout.addWidget(camera_frame)
        
        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Nút điều khiển với icons
        self.setup_control_buttons(control_layout)
        
        left_layout.addWidget(control_panel)
        
        # Panel bên phải
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Text recognition panel
        self.setup_text_panel(right_layout)
        
        # Info panel
        self.setup_info_panel(right_layout)
        
        # Thêm panels vào splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        main_layout.addWidget(splitter)
        
        # Thanh trạng thái
        self.statusBar().showMessage("Sẵn sàng")
        
    def setup_toolbar_actions(self, toolbar):
        # Settings action
        settings_action = QAction(QIcon("icons/settings.png"), "Cài đặt", self)
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)
        
        # Help action
        help_action = QAction(QIcon("icons/help.png"), "Hướng dẫn", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
        
        # About action
        about_action = QAction(QIcon("icons/about.png"), "Thông tin", self)
        about_action.triggered.connect(self.show_about)
        toolbar.addAction(about_action)

    def setup_control_buttons(self, layout):
        button_style = """
            QPushButton {
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 130px;
            }
            QPushButton:checked {
                background-color: #f44336;
            }
            QPushButton[active=true] {
                background-color: #4CAF50;
            }
        """
        
        # Camera control
        self.start_button = QPushButton("◉ Bắt đầu Camera")
        self.start_button.setCheckable(True)
        self.start_button.setStyleSheet(button_style)
        self.start_button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.start_button)
        
        # Landmarks toggle
        self.landmark_button = QPushButton("☐ Hiển thị Landmarks")
        self.landmark_button.setCheckable(True)
        self.landmark_button.setChecked(True)
        self.landmark_button.setStyleSheet(button_style)
        self.landmark_button.clicked.connect(self.toggle_landmarks)
        layout.addWidget(self.landmark_button)
        
        # Đổi nút xác suất thành nút âm thanh
        self.audio_button = QPushButton("☐ Âm thanh")
        self.audio_button.setCheckable(True)
        self.audio_button.setChecked(False)
        self.audio_button.setStyleSheet(button_style)
        self.audio_button.clicked.connect(self.toggle_audio)
        layout.addWidget(self.audio_button)

    def toggle_audio(self):
        self.is_auto_speak = not self.is_auto_speak
        btn_text = "☒ Âm thanh BẬT" if self.is_auto_speak else "☐ Âm thanh"
        self.audio_button.setText(btn_text)
        self.statusBar().showMessage(
            f"{'Đã bật' if self.is_auto_speak else 'Đã tắt'} chức năng đọc văn bản"
        )

    def setup_text_panel(self, layout):
        text_group = QGroupBox("Văn bản nhận dạng")
        text_layout = QVBoxLayout()
        
        # Text display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 8px;
                padding: 10px;
                font-size: 18px;
                line-height: 1.5;
            }
        """)
        text_layout.addWidget(self.text_display)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        # Clear button
        clear_button = QPushButton("Xóa văn bản")
        clear_button.clicked.connect(self.clear_text)
        button_layout.addWidget(clear_button)
        
        text_layout.addLayout(button_layout)
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)

    def speak_text(self):
        """Đọc văn bản hiện tại"""
        if self.is_speaking:
            return
            
        text = self.text_display.toPlainText().strip()
        if not text:
            return
            
        self.is_speaking = True
        threading.Thread(target=self._speak_thread, args=(text,)).start()

    def _speak_thread(self, text):
        """Thread xử lý text-to-speech với xử lý lỗi tốt hơn"""
        try:
            if text == self.last_spoken_text:
                return
                
            self.last_spoken_text = text
            print(f"Attempting to speak: {text}")
            
            # Đảm bảo thư mục Voice tồn tại
            if not os.path.exists(self.voice_dir):
                os.makedirs(self.voice_dir)
            
            # Tạo tên file an toàn không có ký tự đặc biệt
            timestamp = str(int(time.time() * 1000))
            safe_filename = f'speech_{timestamp}.mp3'
            audio_path = os.path.join(self.voice_dir, safe_filename)
            
            print(f"Saving audio to: {audio_path}")
            
            # Lưu file âm thanh
            tts = gtts.gTTS(text=text, lang='vi')
            tts.save(audio_path)
            
            # Đảm bảo file đã được tạo
            if not os.path.exists(audio_path):
                raise Exception("Audio file was not created")
                
            time.sleep(0.1)  # Đợi file được ghi xong
            
            # Phát âm thanh
            try:
                mixer.music.load(audio_path)
                mixer.music.play()
                while mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Pygame mixer error: {e}")
                self.statusBar().showMessage(f"Lỗi phát âm thanh: {str(e)}")
            finally:
                mixer.music.unload()
                
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            self.statusBar().showMessage(f"Lỗi text-to-speech: {str(e)}")
        finally:
            self.is_speaking = False
            # Dọn dẹp file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                print(f"Error removing audio file: {e}")

    def load_model_and_actions(self):
        try:
            # Load model
            model_path = os.path.join('Models', 'final_model.keras')
            if not os.path.exists(model_path):
                QMessageBox.critical(self, "Lỗi", "Không tìm thấy model!")
                return

            self.model = tf.keras.models.load_model(model_path)
            
            # Load mapping
            mapping_file = os.path.join('Logs', 'action_mapping.json')
            if os.path.exists(mapping_file):
                try:
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                        if isinstance(mapping_data, dict) and 'actions' in mapping_data:
                            self.action_mapping = mapping_data['actions']
                            self.actions = sorted(list(self.action_mapping.keys()))
                            self.statusBar().showMessage("Đã tải mapping thành công")
                except Exception as e:
                    print(f"Lỗi khi tải mapping: {str(e)}")
                    self.action_mapping = None
            
            # Nếu không có mapping, load trực tiếp từ thư mục
            if not self.action_mapping:
                data_dir = os.path.join('Data')
                if not os.path.exists(data_dir):
                    QMessageBox.critical(self, "Lỗi", "Không tìm thấy thư mục Data!")
                    return
                
                self.actions = sorted(os.listdir(data_dir))
                self.action_mapping = {action: action for action in self.actions}
            
            # Hiển thị danh sách với tên gốc
            for processed_name, original_name in self.action_mapping.items():
                item = QListWidgetItem(f"{original_name} ({processed_name})")
                self.action_list.addItem(item)
            
            self.statusBar().showMessage("Đã tải model và danh sách hành động")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi tải model: {str(e)}")
    
    def toggle_camera(self):
        if self.timer is None:
            # Bắt đầu camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Lỗi", "Không thể kết nối camera!")
                self.start_button.setChecked(False)
                return
                
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.is_camera_on = True
            self.start_button.setText("⬤ Dừng Camera")
            self.statusBar().showMessage("Camera đang hoạt động")
        else:
            # Dừng camera
            self.timer.stop()
            self.timer = None
            if self.cap:
                self.cap.release()
            self.cap = None
            self.camera_label.clear()
            self.is_camera_on = False
            self.start_button.setText("◉ Bắt đầu Camera")
            self.statusBar().showMessage("Camera đã dừng")
    
    def toggle_landmarks(self):
        self.show_landmarks = not self.show_landmarks
        self.landmark_button.setText("☒ Đang hiện Landmarks" if self.show_landmarks else "☐ Hiển thị Landmarks")
        self.statusBar().showMessage(f"{'Hiện' if self.show_landmarks else 'Ẩn'} landmarks")
    
    def toggle_recording(self):
        self.is_recording = not self.is_recording
        self.record_button.setText("⬤ Đang Ghi" if self.is_recording else "◉ Bắt đầu Ghi")
        self.record_button.setStyleSheet("""
            background-color: #f44336;
        """ if self.is_recording else """
            background-color: #2196F3;
        """)
        self.statusBar().showMessage(f"{'Đang ghi' if self.is_recording else 'Dừng ghi'}")
    
    def update_frame(self):
        if self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Xử lý frame
        image, results = self.mediapipe_detection(frame)
        
        if self.show_landmarks:
            self.draw_landmarks(image, results)
        
        # Nhận dạng
        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints)
        
        if len(self.sequence) == 60:
            self.process_prediction(image)
        
        # Hiển thị frame
        self.display_frame(image)
    
    def mediapipe_detection(self, image):
        """Xử lý ảnh với MediaPipe"""
        image = cv2.flip(image, 1)
        h, w = image.shape[:2]
        self.image_height, self.image_width = h, w
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Provide image dimensions to MediaPipe
        results = self.holistic.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(self, image, results):
        """Vẽ landmarks MediaPipe"""
        # Draw face landmarks
        self.mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_TESSELATION,
            self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
        
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
        
        # Draw hand landmarks
        for hand_landmarks, color_base in [
            (results.left_hand_landmarks, (121,22,76)), 
            (results.right_hand_landmarks, (245,117,66))
        ]:
            if hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=color_base, thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(color_base[0], color_base[1]+22, color_base[2]+164), thickness=2, circle_radius=2)
                )

    def extract_keypoints(self, results):
        """Trích xuất keypoints khớp với mô hình (126 features)"""
        # Pose: chỉ lấy x, y (bỏ z) cho 33 điểm = 66 features
        pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*2)
        
        # Hands: chỉ lấy 15 điểm quan trọng mỗi tay
        important_hand_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        
        # Left hand
        if results.left_hand_landmarks:
            lh = np.array([[results.left_hand_landmarks.landmark[idx].x,
                            results.left_hand_landmarks.landmark[idx].y] 
                           for idx in important_hand_indices]).flatten()
        else:
            lh = np.zeros(30)
            
        # Right hand
        if results.right_hand_landmarks:
            rh = np.array([[results.right_hand_landmarks.landmark[idx].x,
                            results.right_hand_landmarks.landmark[idx].y] 
                           for idx in important_hand_indices]).flatten()
        else:
            rh = np.zeros(30)
        
        return np.concatenate([pose, lh, rh])

    def process_prediction(self, image):
        input_data = np.expand_dims(list(self.sequence), axis=0)
        pred = self.model.predict(input_data, verbose=0)[0]
        self.predictions.append(pred)
        avg_pred = np.mean(list(self.predictions), axis=0)
        
        if max(avg_pred) > 0.7:
            pred_idx = np.argmax(avg_pred)
            processed_action = self.actions[pred_idx]
            original_action = self.action_mapping.get(processed_action, processed_action)
            confidence = avg_pred[pred_idx]
            
            # Cập nhật UI với tên gốc
            self.prediction_label.setText(f"Nhận dạng: {original_action}")
            self.confidence_label.setText(f"Độ tin cậy: {confidence:.2%}")
            
            # Kiểm tra dự đoán trùng lặp
            if processed_action == self.last_prediction:
                self.consecutive_count += 1
                if self.consecutive_count == self.min_consecutive:
                    # Thêm từ mới vào câu với tên gốc
                    original_text = original_action.replace("_", " ")  # Thay thế gạch dưới bằng dấu cách
                    self.current_sentence.append(original_text)
                    
                    # Cập nhật text display
                    cursor = self.text_display.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    self.text_display.setTextCursor(cursor)
                    self.text_display.insertPlainText(f"{original_text} ")
                    
                    # Đọc ngay khi âm thanh được bật
                    if self.is_auto_speak:
                        print("Auto-speak triggered")  # Debug logging
                        self.speak_text()  # Call speak_text directly
            else:
                self.consecutive_count = 1
                self.last_prediction = processed_action
            
            # Highlight action trong list với tên gốc
            for i in range(self.action_list.count()):
                item = self.action_list.item(i)
                if processed_action in item.text():  # Kiểm tra cả tên đã xử lý
                    item.setBackground(QColor(0, 255, 0, 100))
                else:
                    item.setBackground(QColor(255, 255, 255))

    def display_frame(self, image):
        # Chuyển đổi frame sang định dạng QImage
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label size
        scaled_image = qt_image.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
        self.camera_label.setPixmap(QPixmap.fromImage(scaled_image))

    def clear_text(self):
        """Xóa nội dung văn bản và reset trạng thái"""
        self.text_display.clear()
        self.current_sentence = []
        self.last_spoken_text = ""

    def show_settings(self):
        """Hiển thị dialog cài đặt"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Cài đặt")
        layout = QVBoxLayout(dialog)
        
        # Threshold settings
        threshold_group = QGroupBox("Ngưỡng nhận dạng")
        threshold_layout = QFormLayout()
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(95)
        self.threshold_slider.setValue(int(self.min_consecutive * 10))
        threshold_layout.addRow("Số lần lặp tối thiểu:", self.threshold_slider)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # Auto-speak settings
        speak_group = QGroupBox("Cài đặt đọc tự động")
        speak_layout = QFormLayout()
        
        self.words_slider = QSlider(Qt.Horizontal)
        self.words_slider.setMinimum(1)
        self.words_slider.setMaximum(10)
        self.words_slider.setValue(self.auto_speak_threshold)
        speak_layout.addRow("Số từ tối thiểu:", self.words_slider)
        
        speak_group.setLayout(speak_layout)
        layout.addWidget(speak_group)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            self.min_consecutive = self.threshold_slider.value() / 10
            self.auto_speak_threshold = self.words_slider.value()

    def show_help(self):
        """Hiển thị hướng dẫn sử dụng"""
        QMessageBox.information(self, "Hướng dẫn sử dụng",
            "1. Bấm 'Bắt đầu Camera' để khởi động nhận dạng\n"
            "2. Sử dụng các nút điều khiển để bật/tắt tính năng\n"
            "3. Văn bản nhận dạng sẽ hiển thị ở khung bên phải\n"
            "4. Có thể bật chế độ tự động đọc văn bản\n"
            "5. Điều chỉnh các cài đặt trong menu Settings"
        )

    def show_about(self):
        """Hiển thị thông tin về ứng dụng"""
        QMessageBox.about(self, "Thông tin",
            "Hệ thống nhận dạng ngôn ngữ ký hiệu\n"
            "Phiên bản: 1.0\n"
            "© 2024 Bản quyền thuộc về nhóm phát triển"
        )

    def closeEvent(self, event):
        """Cleanup khi đóng ứng dụng"""
        if self.cap:
            self.cap.release()
        
        # Cleanup audio
        try:
            mixer.quit()
        except:
            pass
        
        # Cleanup voice directory
        if os.path.exists(self.voice_dir):
            try:
                for file in os.listdir(self.voice_dir):
                    file_path = os.path.join(self.voice_dir, file)
                    if file.startswith('speech_') and file.endswith('.mp3'):
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
            except Exception as e:
                print(f"Error cleaning up voice directory: {e}")
                
        event.accept()

    def setup_info_panel(self, layout):
        """Thiết lập panel thông tin nhận dạng"""
        info_group = QGroupBox("Thông Tin Nhận Dạng")
        info_layout = QVBoxLayout()
        
        # Nhãn hiển thị kết quả nhận dạng
        self.prediction_label = QLabel("Đang chờ...")
        self.prediction_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2196F3;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 5px;
            }
        """)
        info_layout.addWidget(self.prediction_label)
        
        # Nhãn hiển thị độ tin cậy
        self.confidence_label = QLabel("Độ tin cậy: --")
        self.confidence_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #666;
                padding: 5px;
            }
        """)
        info_layout.addWidget(self.confidence_label)
        
        # Danh sách các hành động
        action_label = QLabel("Danh sách hành động:")
        action_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        info_layout.addWidget(action_label)
        
        self.action_list = QListWidget()
        self.action_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976D2;
            }
        """)
        self.action_list.setMinimumHeight(200)
        info_layout.addWidget(self.action_list)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Sử dụng style Fusion cho giao diện đẹp hơn
    
    # Set stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            min-width: 100px;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #0D47A1;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 6px;
            margin-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QListWidget {
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
        QTextEdit {
            background-color: white;
            color: #333333;
            font-size: 16px;
        }
        
        QPushButton#clear_button {
            background-color: #f44336;
        }
        
        QPushButton#clear_button:hover {
            background-color: #d32f2f;
        }
    """)
    
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())


