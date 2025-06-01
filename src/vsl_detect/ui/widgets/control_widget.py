"""
Control widget for Vietnamese Sign Language Detection.
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ...utils.config import Config
from ...utils.logger import get_logger


class ControlWidget(QWidget):
    """Widget for controlling application functions."""
    
    # Signals
    camera_toggled = pyqtSignal(bool)
    landmarks_toggled = pyqtSignal(bool)
    audio_toggled = pyqtSignal(bool)
    reset_clicked = pyqtSignal()
    
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.logger = get_logger(__name__)
        
        # Control state
        self._camera_enabled = False
        self._landmarks_enabled = config.ui.show_landmarks
        self._audio_enabled = config.audio.auto_speak
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup control widget UI."""
        layout = QHBoxLayout(self)
        
        # Main controls group
        main_group = QGroupBox("Điều khiển chính")
        main_layout = QHBoxLayout(main_group)
        
        # Camera control
        self.camera_button = QPushButton("📹 Bắt đầu Camera")
        self.camera_button.setCheckable(True)
        self.camera_button.setToolTip("Bật/tắt camera")
        self.camera_button.clicked.connect(self._on_camera_clicked)
        main_layout.addWidget(self.camera_button)
        
        # Landmarks control
        self.landmarks_button = QPushButton("👤 Hiển thị Landmarks")
        self.landmarks_button.setCheckable(True)
        self.landmarks_button.setChecked(self._landmarks_enabled)
        self.landmarks_button.setToolTip("Hiển thị/ẩn landmarks MediaPipe")
        self.landmarks_button.clicked.connect(self._on_landmarks_clicked)
        main_layout.addWidget(self.landmarks_button)
        
        # Audio control
        self.audio_button = QPushButton("🔊 Âm thanh")
        self.audio_button.setCheckable(True)
        self.audio_button.setChecked(self._audio_enabled)
        self.audio_button.setToolTip("Bật/tắt đọc văn bản tự động")
        self.audio_button.clicked.connect(self._on_audio_clicked)
        main_layout.addWidget(self.audio_button)
        
        # Reset button
        self.reset_button = QPushButton("🔄 Reset")
        self.reset_button.setToolTip("Reset trạng thái nhận dạng")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        main_layout.addWidget(self.reset_button)
        
        layout.addWidget(main_group)
        
        # Advanced controls group
        advanced_group = QGroupBox("Cài đặt nâng cao")
        advanced_layout = QHBoxLayout(advanced_group)
        
        # Confidence threshold
        confidence_layout = QVBoxLayout()
        confidence_layout.addWidget(QLabel("Độ tin cậy:"))
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(50)
        self.confidence_slider.setMaximum(95)
        self.confidence_slider.setValue(int(self.config.model.prediction_threshold * 100))
        self.confidence_slider.setToolTip("Ngưỡng độ tin cậy tối thiểu")
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)
        confidence_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel(f"{self.config.model.prediction_threshold:.2f}")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        confidence_layout.addWidget(self.confidence_label)
        
        advanced_layout.addLayout(confidence_layout)
        
        # Prediction smoothing
        smoothing_layout = QVBoxLayout()
        smoothing_layout.addWidget(QLabel("Làm mượt:"))
        
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setMinimum(1)
        self.smoothing_slider.setMaximum(10)
        self.smoothing_slider.setValue(self.config.model.prediction_smoothing_window)
        self.smoothing_slider.setToolTip("Số khung hình để làm mượt dự đoán")
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_changed)
        smoothing_layout.addWidget(self.smoothing_slider)
        
        self.smoothing_label = QLabel(str(self.config.model.prediction_smoothing_window))
        self.smoothing_label.setAlignment(Qt.AlignCenter)
        smoothing_layout.addWidget(self.smoothing_label)
        
        advanced_layout.addLayout(smoothing_layout)
        
        layout.addWidget(advanced_group)
        
        # Apply styling
        self._apply_styling()
    
    def _apply_styling(self) -> None:
        """Apply widget styling."""
        button_style = """
            QPushButton {
                padding: 10px 15px;
                border: 2px solid #ddd;
                border-radius: 6px;
                background-color: #f8f9fa;
                font-size: 13px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton:checked {
                background-color: #28a745;
                border-color: #1e7e34;
                color: white;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """
        
        reset_style = """
            QPushButton {
                padding: 10px 15px;
                border: 2px solid #dc3545;
                border-radius: 6px;
                background-color: #f8f9fa;
                color: #dc3545;
                font-size: 13px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #dc3545;
                color: white;
            }
            QPushButton:pressed {
                background-color: #c82333;
            }
        """
        
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #d4d4d4, stop:1 #afafaf);
            }
        """
        
        # Apply styles
        for button in [self.camera_button, self.landmarks_button, self.audio_button]:
            button.setStyleSheet(button_style)
        
        self.reset_button.setStyleSheet(reset_style)
        
        for slider in [self.confidence_slider, self.smoothing_slider]:
            slider.setStyleSheet(slider_style)
    
    def _on_camera_clicked(self, checked: bool) -> None:
        """Handle camera button click."""
        self._camera_enabled = checked
        
        if checked:
            self.camera_button.setText("📹 Dừng Camera")
            self.camera_button.setToolTip("Dừng camera")
        else:
            self.camera_button.setText("📹 Bắt đầu Camera")
            self.camera_button.setToolTip("Bắt đầu camera")
        
        self.camera_toggled.emit(checked)
        self.logger.info(f"Camera {'enabled' if checked else 'disabled'}")
    
    def _on_landmarks_clicked(self, checked: bool) -> None:
        """Handle landmarks button click."""
        self._landmarks_enabled = checked
        
        if checked:
            self.landmarks_button.setText("👤 Ẩn Landmarks")
            self.landmarks_button.setToolTip("Ẩn landmarks MediaPipe")
        else:
            self.landmarks_button.setText("👤 Hiển thị Landmarks")
            self.landmarks_button.setToolTip("Hiển thị landmarks MediaPipe")
        
        self.landmarks_toggled.emit(checked)
        self.logger.info(f"Landmarks {'enabled' if checked else 'disabled'}")
    
    def _on_audio_clicked(self, checked: bool) -> None:
        """Handle audio button click."""
        self._audio_enabled = checked
        
        if checked:
            self.audio_button.setText("🔊 Tắt âm thanh")
            self.audio_button.setToolTip("Tắt đọc văn bản tự động")
        else:
            self.audio_button.setText("🔊 Bật âm thanh")
            self.audio_button.setToolTip("Bật đọc văn bản tự động")
        
        self.audio_toggled.emit(checked)
        self.logger.info(f"Audio {'enabled' if checked else 'disabled'}")
    
    def _on_reset_clicked(self) -> None:
        """Handle reset button click."""
        reply = QMessageBox.question(
            self,
            "Xác nhận Reset",
            "Bạn có chắc chắn muốn reset trạng thái nhận dạng?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.reset_clicked.emit()
            self.logger.info("Reset requested")
    
    def _on_confidence_changed(self, value: int) -> None:
        """Handle confidence threshold change."""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        # TODO: Update config or emit signal to update detector
        self.logger.debug(f"Confidence threshold changed to {confidence}")
    
    def _on_smoothing_changed(self, value: int) -> None:
        """Handle smoothing window change."""
        self.smoothing_label.setText(str(value))
        # TODO: Update config or emit signal to update detector
        self.logger.debug(f"Smoothing window changed to {value}")
    
    # Public methods for external control
    def set_camera_state(self, enabled: bool) -> None:
        """Set camera button state externally."""
        self.camera_button.setChecked(enabled)
        self._on_camera_clicked(enabled)
    
    def is_camera_enabled(self) -> bool:
        """Check if camera is enabled."""
        return self._camera_enabled
    
    def is_landmarks_enabled(self) -> bool:
        """Check if landmarks display is enabled."""
        return self._landmarks_enabled
    
    def is_audio_enabled(self) -> bool:
        """Check if audio is enabled."""
        return self._audio_enabled
    
    def get_confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        return self.confidence_slider.value() / 100.0
    
    def get_smoothing_window(self) -> int:
        """Get current smoothing window size."""
        return self.smoothing_slider.value()
    
    def enable_controls(self, enabled: bool) -> None:
        """Enable/disable all controls."""
        for widget in [
            self.landmarks_button, self.audio_button, self.reset_button,
            self.confidence_slider, self.smoothing_slider
        ]:
            widget.setEnabled(enabled)
    
    def set_status_text(self, text: str) -> None:
        """Set status text (if status widget exists)."""
        # This could be expanded to show status in the control widget
        pass