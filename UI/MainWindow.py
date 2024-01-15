"""
Vietnamese Sign Language Detection Main Window
Optimized UI with vertical sidebar
"""

import sys
import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QSlider, QComboBox, QTextEdit,
    QGroupBox, QGridLayout, QSplitter, QFrame, QTabWidget,
    QProgressBar, QCheckBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QFileDialog, QApplication
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon

from Core.Detector import SignLanguageDetector
from Utils.Config import ConfigManager
from Utils.Logger import get_logger

class CameraThread(QThread):
    """Thread for camera processing"""
    frame_ready = pyqtSignal(np.ndarray, dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, detector: SignLanguageDetector, config: Dict[str, Any]):
        super().__init__()
        self.detector = detector
        self.config = config
        self.running = False
        self.camera = None
        
    def run(self):
        """Main camera processing loop"""
        try:
            camera_index = self.config.get('camera_index', 0)
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                self.error_occurred.emit("Không thể mở camera")
                return
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('frame_width', 640))
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('frame_height', 480))
            self.camera.set(cv2.CAP_PROP_FPS, self.config.get('fps', 30))
            
            self.running = True
            
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Flip frame if needed
                if self.config.get('flip_horizontal', True):
                    frame = cv2.flip(frame, 1)
                
                # Process frame
                results = self.detector.process_frame(frame)
                
                # Emit results
                self.frame_ready.emit(results['frame'], results)
                
        except Exception as e:
            self.error_occurred.emit(f"Lỗi camera: {str(e)}")
        finally:
            if self.camera:
                self.camera.release()
    
    def stop(self):
        """Stop camera processing"""
        self.running = False
        self.wait()

class SignLanguageApp(QMainWindow):
    """Main application window with optimized UI"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.detector = None
        self.camera_thread = None
        self.timer = QTimer()
        
        # Setup UI
        self.setup_ui()
        self.setup_styles()
        self.setup_connections()
        
        # Initialize detector
        self.initialize_detector()
        
    def setup_ui(self):
        """Setup main UI components"""
        self.setWindowTitle(self.config.get('window_title', 'Hệ Thống Nhận Dạng Ngôn Ngữ Ký Hiệu'))
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for sidebar and main content
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Create sidebar
        self.create_sidebar(splitter)
        
        # Create main content area
        self.create_main_content(splitter)
        
        # Set splitter proportions
        splitter.setSizes([300, 1100])
        
    def create_sidebar(self, parent):
        """Create vertical sidebar with controls"""
        sidebar = QFrame()
        sidebar.setFrameStyle(QFrame.StyledPanel)
        sidebar.setMaximumWidth(300)
        sidebar.setMinimumWidth(250)
        
        # Sidebar layout
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("ĐIỀU KHIỂN")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        sidebar_layout.addWidget(title_label)
        
        # Camera controls
        self.create_camera_controls(sidebar_layout)
        
        # Detection controls
        self.create_detection_controls(sidebar_layout)
        
        # Model controls
        self.create_model_controls(sidebar_layout)
        
        # Status section
        self.create_status_section(sidebar_layout)
        
        # Add stretch to push everything to top
        sidebar_layout.addStretch()
        
        parent.addWidget(sidebar)
        
    def create_camera_controls(self, layout):
        """Create camera control section"""
        camera_group = QGroupBox("📹 Camera")
        camera_layout = QVBoxLayout(camera_group)
        
        # Camera start/stop button
        self.camera_btn = QPushButton("▶️ Bắt đầu Camera")
        self.camera_btn.setMinimumHeight(40)
        camera_layout.addWidget(self.camera_btn)
        
        # Camera settings
        settings_layout = QGridLayout()
        
        # Camera index
        settings_layout.addWidget(QLabel("Camera:"), 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        settings_layout.addWidget(self.camera_combo, 0, 1)
        
        # Resolution
        settings_layout.addWidget(QLabel("Độ phân giải:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        settings_layout.addWidget(self.resolution_combo, 1, 1)
        
        # FPS
        settings_layout.addWidget(QLabel("FPS:"), 2, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(10, 60)
        self.fps_spin.setValue(30)
        settings_layout.addWidget(self.fps_spin, 2, 1)
        
        camera_layout.addLayout(settings_layout)
        layout.addWidget(camera_group)
        
    def create_detection_controls(self, layout):
        """Create detection control section"""
        detection_group = QGroupBox("🎯 Nhận Dạng")
        detection_layout = QVBoxLayout(detection_group)
        
        # Confidence threshold
        detection_layout.addWidget(QLabel("Ngưỡng tin cậy:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(50, 95)
        self.confidence_slider.setValue(70)
        detection_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("70%")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        detection_layout.addWidget(self.confidence_label)
        
        # Show landmarks checkbox
        self.show_landmarks_cb = QCheckBox("Hiển thị landmarks")
        self.show_landmarks_cb.setChecked(True)
        detection_layout.addWidget(self.show_landmarks_cb)
        
        # Show confidence checkbox
        self.show_confidence_cb = QCheckBox("Hiển thị độ tin cậy")
        self.show_confidence_cb.setChecked(True)
        detection_layout.addWidget(self.show_confidence_cb)
        
        layout.addWidget(detection_group)
        
    def create_model_controls(self, layout):
        """Create model control section"""
        model_group = QGroupBox("🤖 Mô Hình")
        model_layout = QVBoxLayout(model_group)
        
        # Model selection
        model_layout.addWidget(QLabel("Chọn mô hình:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Mô hình mặc định", "Mô hình tùy chỉnh"])
        model_layout.addWidget(self.model_combo)
        
        # Load model button
        self.load_model_btn = QPushButton("📁 Tải Mô Hình")
        self.load_model_btn.setMinimumHeight(35)
        model_layout.addWidget(self.load_model_btn)
        
        # Model info
        self.model_info_label = QLabel("Chưa tải mô hình")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("color: gray; font-size: 10px;")
        model_layout.addWidget(self.model_info_label)
        
        layout.addWidget(model_group)
        
    def create_status_section(self, layout):
        """Create status section"""
        status_group = QGroupBox("📊 Trạng Thái")
        status_layout = QVBoxLayout(status_group)
        
        # Current action
        status_layout.addWidget(QLabel("Hành động hiện tại:"))
        self.current_action_label = QLabel("Chưa phát hiện")
        self.current_action_label.setAlignment(Qt.AlignCenter)
        self.current_action_label.setStyleSheet("font-weight: bold; color: blue;")
        status_layout.addWidget(self.current_action_label)
        
        # Confidence bar
        status_layout.addWidget(QLabel("Độ tin cậy:"))
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        status_layout.addWidget(self.confidence_bar)
        
        # FPS counter
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.fps_label)
        
        layout.addWidget(status_group)
        
    def create_main_content(self, parent):
        """Create main content area"""
        main_content = QWidget()
        main_layout = QVBoxLayout(main_content)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Camera tab
        self.create_camera_tab()
        
        # Logs tab
        self.create_logs_tab()
        
        # Settings tab
        self.create_settings_tab()
        
        parent.addWidget(main_content)
        
    def create_camera_tab(self):
        """Create camera view tab"""
        camera_widget = QWidget()
        camera_layout = QVBoxLayout(camera_widget)
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        camera_layout.addWidget(self.camera_label)
        
        # Camera info
        info_layout = QHBoxLayout()
        self.camera_info_label = QLabel("Camera chưa khởi động")
        info_layout.addWidget(self.camera_info_label)
        info_layout.addStretch()
        
        camera_layout.addLayout(info_layout)
        self.tab_widget.addTab(camera_widget, "📹 Camera")
        
    def create_logs_tab(self):
        """Create logs view tab"""
        logs_widget = QWidget()
        logs_layout = QVBoxLayout(logs_widget)
        
        # Log display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        logs_layout.addWidget(self.log_text)
        
        # Log controls
        log_controls = QHBoxLayout()
        
        self.clear_logs_btn = QPushButton("🗑️ Xóa Logs")
        log_controls.addWidget(self.clear_logs_btn)
        
        self.save_logs_btn = QPushButton("💾 Lưu Logs")
        log_controls.addWidget(self.save_logs_btn)
        
        log_controls.addStretch()
        logs_layout.addLayout(log_controls)
        
        self.tab_widget.addTab(logs_widget, "📋 Logs")
        
    def create_settings_tab(self):
        """Create settings view tab"""
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # Settings grid
        settings_grid = QGridLayout()
        
        # General settings
        settings_grid.addWidget(QLabel("Tự động phát âm:"), 0, 0)
        self.auto_speak_cb = QCheckBox()
        self.auto_speak_cb.setChecked(False)
        settings_grid.addWidget(self.auto_speak_cb, 0, 1)
        
        settings_grid.addWidget(QLabel("Ngôn ngữ:"), 1, 0)
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Tiếng Việt", "English"])
        settings_grid.addWidget(self.language_combo, 1, 1)
        
        settings_grid.addWidget(QLabel("Chủ đề:"), 2, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Mặc định", "Tối", "Sáng"])
        settings_grid.addWidget(self.theme_combo, 2, 1)
        
        settings_layout.addLayout(settings_grid)
        settings_layout.addStretch()
        
        self.tab_widget.addTab(settings_widget, "⚙️ Cài Đặt")
        
    def setup_styles(self):
        """Setup application styles"""
        # Modern dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4a4a4a;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 8px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QComboBox {
                background-color: #4a4a4a;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #4a4a4a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                background-color: #4a4a4a;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #4a4a4a;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
            }
        """)
        
    def setup_connections(self):
        """Setup signal connections"""
        # Camera controls
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        
        # Model controls
        self.load_model_btn.clicked.connect(self.load_model)
        
        # Log controls
        self.clear_logs_btn.clicked.connect(self.clear_logs)
        self.save_logs_btn.clicked.connect(self.save_logs)
        
        # Timer for FPS update
        self.timer.timeout.connect(self.update_fps)
        self.timer.start(1000)  # Update every second
        
    def initialize_detector(self):
        """Initialize the sign language detector"""
        try:
            self.detector = SignLanguageDetector(self.config)
            self.logger.info("Detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể khởi tạo detector: {e}")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.camera_thread is None or not self.camera_thread.running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera processing"""
        try:
            # Update config with current settings
            self.config['camera_index'] = self.camera_combo.currentIndex()
            self.config['prediction_threshold'] = self.confidence_slider.value() / 100.0
            
            # Create and start camera thread
            self.camera_thread = CameraThread(self.detector, self.config)
            self.camera_thread.frame_ready.connect(self.update_camera_frame)
            self.camera_thread.error_occurred.connect(self.handle_camera_error)
            self.camera_thread.start()
            
            # Update UI
            self.camera_btn.setText("⏹️ Dừng Camera")
            self.camera_info_label.setText("Camera đang chạy")
            
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể khởi động camera: {e}")
    
    def stop_camera(self):
        """Stop camera processing"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        # Update UI
        self.camera_btn.setText("▶️ Bắt đầu Camera")
        self.camera_info_label.setText("Camera đã dừng")
        self.camera_label.clear()
    
    def update_camera_frame(self, frame: np.ndarray, results: Dict[str, Any]):
        """Update camera frame display"""
        try:
            # Convert frame to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Scale image to fit label
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.camera_label.setPixmap(scaled_pixmap)
            
            # Update detection results
            self.update_detection_results(results)
            
        except Exception as e:
            self.logger.error(f"Error updating camera frame: {e}")
    
    def update_detection_results(self, results: Dict[str, Any]):
        """Update detection results in UI"""
        action = results.get('action', 'Chưa phát hiện')
        confidence = results.get('confidence', 0.0)
        
        # Update action label
        self.current_action_label.setText(action)
        
        # Update confidence bar
        confidence_percent = int(confidence * 100)
        self.confidence_bar.setValue(confidence_percent)
        
        # Update log
        if action != 'Chưa phát hiện' and confidence > 0.7:
            self.log_message(f"Phát hiện: {action} (Độ tin cậy: {confidence_percent}%)")
    
    def update_confidence_label(self, value: int):
        """Update confidence label when slider changes"""
        self.confidence_label.setText(f"{value}%")
        if self.detector:
            self.detector.confidence_threshold = value / 100.0
    
    def load_model(self):
        """Load custom model"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Chọn mô hình", "", 
                "Model files (*.keras *.h5 *.pkl *.joblib);;All files (*)"
            )
            
            if file_path:
                success = self.detector.load_model(file_path)
                if success:
                    self.model_info_label.setText(f"Đã tải: {Path(file_path).name}")
                    self.log_message(f"Đã tải mô hình: {file_path}")
                else:
                    QMessageBox.warning(self, "Cảnh báo", "Không thể tải mô hình")
                    
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi tải mô hình: {e}")
    
    def log_message(self, message: str):
        """Add message to log display"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.clear()
    
    def save_logs(self):
        """Save logs to file"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Lưu logs", "", "Text files (*.txt);;All files (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"Đã lưu logs: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving logs: {e}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi lưu logs: {e}")
    
    def update_fps(self):
        """Update FPS display"""
        if self.camera_thread and self.camera_thread.running:
            # Simple FPS calculation (you can implement more sophisticated FPS tracking)
            self.fps_label.setText("FPS: 30")
        else:
            self.fps_label.setText("FPS: 0")
    
    def handle_camera_error(self, error_message: str):
        """Handle camera errors"""
        self.logger.error(f"Camera error: {error_message}")
        QMessageBox.critical(self, "Lỗi Camera", error_message)
        self.stop_camera()
    
    def closeEvent(self, event):
        """Handle application close event"""
        self.stop_camera()
        if self.detector:
            self.detector.cleanup()
        event.accept()
