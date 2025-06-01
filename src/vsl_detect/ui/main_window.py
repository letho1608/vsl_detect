"""
Main application window for Vietnamese Sign Language Detection.
"""

import cv2
import sys
import numpy as np
from typing import Optional

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ..core.detector import SignLanguageDetector, DetectionResult
from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.audio import AudioManager, AutoSpeaker
from .widgets.camera_widget import CameraWidget
from .widgets.text_widget import TextWidget
from .widgets.control_widget import ControlWidget


class SignLanguageApp(QMainWindow):
    """Main application window."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        
        # Core components
        self.detector: Optional[SignLanguageDetector] = None
        self.audio_manager: Optional[AudioManager] = None
        self.auto_speaker: Optional[AutoSpeaker] = None
        
        # UI components
        self.camera_widget: Optional[CameraWidget] = None
        self.text_widget: Optional[TextWidget] = None
        self.control_widget: Optional[ControlWidget] = None
        
        # Camera state
        self.cap: Optional[cv2.VideoCapture] = None
        self.timer: Optional[QTimer] = None
        self.is_camera_active = False
        
        # Initialize application
        self._setup_window()
        self._setup_ui()
        self._initialize_components()
        
        self.logger.info("Main application window initialized")
    
    def _setup_window(self) -> None:
        """Setup main window properties."""
        self.setWindowTitle(self.config.ui.window_title)
        self.setGeometry(100, 100, self.config.ui.window_width, self.config.ui.window_height)
        
        # Set window icon if available
        try:
            icon_path = self.config.get_absolute_path("src/vsl_detect/ui/icons/app.png")
            if QFile.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except:
            pass  # Ignore if icon not found
    
    def _setup_ui(self) -> None:
        """Setup user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self._create_toolbar()
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (Camera)
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel (Text and controls)
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Sẵn sàng")
    
    def _create_toolbar(self) -> None:
        """Create application toolbar."""
        toolbar = self.addToolBar("Main")
        
        # Settings action
        settings_action = QAction("⚙️ Cài đặt", self)
        settings_action.triggered.connect(self._show_settings)
        toolbar.addAction(settings_action)
        
        # Help action
        help_action = QAction("❓ Hướng dẫn", self)
        help_action.triggered.connect(self._show_help)
        toolbar.addAction(help_action)
        
        # About action
        about_action = QAction("ℹ️ Thông tin", self)
        about_action.triggered.connect(self._show_about)
        toolbar.addAction(about_action)
    
    def _create_left_panel(self) -> QWidget:
        """Create left panel with camera."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Camera widget
        self.camera_widget = CameraWidget(self.config)
        layout.addWidget(self.camera_widget)
        
        # Control widget
        self.control_widget = ControlWidget(self.config)
        layout.addWidget(self.control_widget)
        
        # Connect control signals
        self.control_widget.camera_toggled.connect(self._toggle_camera)
        self.control_widget.landmarks_toggled.connect(self._toggle_landmarks)
        self.control_widget.audio_toggled.connect(self._toggle_audio)
        self.control_widget.reset_clicked.connect(self._reset_detection)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create right panel with text and info."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Text widget
        self.text_widget = TextWidget(self.config)
        layout.addWidget(self.text_widget)
        
        # Action list
        actions_group = QGroupBox("Các hành động có thể nhận dạng")
        actions_layout = QVBoxLayout(actions_group)
        
        self.action_list = QListWidget()
        self.action_list.setMaximumHeight(200)
        actions_layout.addWidget(self.action_list)
        
        layout.addWidget(actions_group)
        
        # Info panel
        info_group = QGroupBox("Thông tin hệ thống")
        info_layout = QVBoxLayout(info_group)
        
        self.info_label = QLabel("Chưa khởi tạo")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        
        layout.addWidget(info_group)
        
        # Connect text widget signals
        self.text_widget.speak_requested.connect(self._speak_text)
        self.text_widget.clear_requested.connect(self._clear_text)
        
        return panel
    
    def _initialize_components(self) -> None:
        """Initialize core components."""
        try:
            # Initialize detector
            self.detector = SignLanguageDetector(self.config)
            
            # Initialize audio
            self.audio_manager = AudioManager(self.config)
            self.auto_speaker = AutoSpeaker(self.audio_manager, self.config)
            
            # Populate action list
            actions = self.detector.get_actions()
            for action in actions:
                self.action_list.addItem(action)
            
            # Update info
            self._update_info()
            
            self.statusBar().showMessage("Hệ thống đã sẵn sàng")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            QMessageBox.critical(self, "Lỗi Khởi Tạo", f"Không thể khởi tạo hệ thống:\n{str(e)}")
    
    def _toggle_camera(self, enabled: bool) -> None:
        """Toggle camera on/off."""
        if enabled and not self.is_camera_active:
            self._start_camera()
        elif not enabled and self.is_camera_active:
            self._stop_camera()
    
    def _start_camera(self) -> None:
        """Start camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.config.camera.camera_index)
            
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
            
            # Start timer for frame updates
            self.timer = QTimer()
            self.timer.timeout.connect(self._update_frame)
            self.timer.start(33)  # ~30 FPS
            
            self.is_camera_active = True
            self.statusBar().showMessage("Camera đang hoạt động")
            self.logger.info("Camera started")
            
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            QMessageBox.critical(self, "Lỗi Camera", f"Không thể khởi động camera:\n{str(e)}")
            self.control_widget.set_camera_state(False)
    
    def _stop_camera(self) -> None:
        """Stop camera capture."""
        if self.timer:
            self.timer.stop()
            self.timer = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_widget.clear_image()
        self.is_camera_active = False
        self.statusBar().showMessage("Camera đã dừng")
        self.logger.info("Camera stopped")
    
    def _update_frame(self) -> None:
        """Update frame from camera."""
        if not self.cap or not self.detector:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        try:
            # Process frame with detector
            show_landmarks = self.control_widget.is_landmarks_enabled()
            result = self.detector.process_frame(frame, draw_landmarks=show_landmarks)
            
            # Display frame
            self.camera_widget.display_frame(frame)
            
            # Handle detection result
            if result.text:
                self._handle_detection(result)
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
    
    def _handle_detection(self, result: DetectionResult) -> None:
        """Handle detection result."""
        # Add to text widget
        self.text_widget.add_text(result.text)
        
        # Add to auto speaker if enabled
        if self.control_widget.is_audio_enabled():
            self.auto_speaker.add_word(result.text)
        
        # Update status
        self.statusBar().showMessage(f"Nhận dạng: {result.text} ({result.confidence:.2f})")
    
    def _toggle_landmarks(self, enabled: bool) -> None:
        """Toggle landmark display."""
        self.logger.info(f"Landmarks {'enabled' if enabled else 'disabled'}")
    
    def _toggle_audio(self, enabled: bool) -> None:
        """Toggle audio output."""
        if self.auto_speaker:
            self.auto_speaker.set_auto_speak(enabled)
        self.logger.info(f"Audio {'enabled' if enabled else 'disabled'}")
    
    def _reset_detection(self) -> None:
        """Reset detection state."""
        if self.detector:
            self.detector.reset_sequence()
        if self.auto_speaker:
            self.auto_speaker.clear_sentence()
        self.statusBar().showMessage("Đã reset hệ thống nhận dạng")
    
    def _speak_text(self, text: str) -> None:
        """Speak given text."""
        if self.audio_manager:
            self.audio_manager.speak_text(text)
    
    def _clear_text(self) -> None:
        """Clear text display."""
        if self.auto_speaker:
            self.auto_speaker.clear_sentence()
    
    def _update_info(self) -> None:
        """Update system information display."""
        if not self.detector:
            return
        
        info_text = f"""
        <b>Thông tin hệ thống:</b><br>
        • Model: {self.config.model.model_path}<br>
        • Số hành động: {len(self.detector.get_actions())}<br>
        • Độ tin cậy tối thiểu: {self.config.model.prediction_threshold}<br>
        • Cửa sổ dự đoán: {self.config.model.sequence_length} khung hình<br>
        • Camera: {self.config.camera.camera_index}<br>
        • Ngôn ngữ TTS: {self.config.audio.language}
        """
        self.info_label.setText(info_text)
    
    def _show_settings(self) -> None:
        """Show settings dialog."""
        # TODO: Implement settings dialog
        QMessageBox.information(self, "Cài đặt", "Tính năng cài đặt sẽ được thêm trong phiên bản tiếp theo.")
    
    def _show_help(self) -> None:
        """Show help dialog."""
        help_text = """
        <h3>Hướng dẫn sử dụng</h3>
        <p><b>1. Bắt đầu camera:</b> Nhấn nút "Bắt đầu Camera"</p>
        <p><b>2. Thực hiện ký hiệu:</b> Thực hiện các ký hiệu trước camera</p>
        <p><b>3. Xem kết quả:</b> Văn bản được nhận dạng sẽ hiển thị bên phải</p>
        <p><b>4. Nghe âm thanh:</b> Bật "Âm thanh" để nghe văn bản được đọc</p>
        <p><b>5. Reset:</b> Nhấn "Reset" để xóa trạng thái nhận dạng</p>
        """
        QMessageBox.information(self, "Hướng dẫn", help_text)
    
    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """
        <h3>Vietnamese Sign Language Detection System</h3>
        <p>Phiên bản: 1.0.0</p>
        <p>Hệ thống nhận dạng ngôn ngữ ký hiệu Việt Nam sử dụng Deep Learning</p>
        <p><b>Công nghệ:</b></p>
        <ul>
        <li>TensorFlow 2.18.0</li>
        <li>MediaPipe 0.10.20</li>
        <li>PyQt5 5.15.11</li>
        <li>OpenCV</li>
        </ul>
        <p>© 2024 Vietnamese Sign Language Team</p>
        """
        QMessageBox.about(self, "Thông tin", about_text)
    
    def cleanup(self) -> None:
        """Cleanup resources before closing."""
        self.logger.info("Cleaning up application")
        
        # Stop camera
        self._stop_camera()
        
        # Cleanup components
        if self.detector:
            self.detector.cleanup()
        
        if self.audio_manager:
            self.audio_manager.cleanup()
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        self.cleanup()
        event.accept()