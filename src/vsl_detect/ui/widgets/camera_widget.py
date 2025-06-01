"""
Camera display widget for Vietnamese Sign Language Detection.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ...utils.config import Config
from ...utils.logger import get_logger


class CameraWidget(QLabel):
    """Widget for displaying camera feed."""
    
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.logger = get_logger(__name__)
        
        # Setup widget
        self._setup_widget()
        
        # Display placeholder
        self._show_placeholder()
    
    def _setup_widget(self) -> None:
        """Setup widget properties."""
        self.setMinimumSize(640, 480)
        self.setMaximumSize(1280, 960)
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignCenter)
        
        # Styling
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 10px;
                background-color: #f0f0f0;
                color: #666666;
                font-size: 14px;
            }
        """)
    
    def _show_placeholder(self) -> None:
        """Show placeholder when no camera feed."""
        placeholder_pixmap = QPixmap(640, 480)
        placeholder_pixmap.fill(Qt.lightGray)
        
        painter = QPainter(placeholder_pixmap)
        painter.setPen(Qt.darkGray)
        painter.setFont(QFont("Arial", 16))
        painter.drawText(
            placeholder_pixmap.rect(),
            Qt.AlignCenter,
            "📹\nChưa có camera\nNhấn 'Bắt đầu Camera' để bắt đầu"
        )
        painter.end()
        
        self.setPixmap(placeholder_pixmap)
    
    def display_frame(self, frame: np.ndarray) -> None:
        """
        Display video frame in the widget.
        
        Args:
            frame: OpenCV frame (BGR format)
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get frame dimensions
            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width
            
            # Create QImage
            q_image = QImage(
                rgb_frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888
            )
            
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit widget while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"Error displaying frame: {e}")
    
    def clear_image(self) -> None:
        """Clear the displayed image and show placeholder."""
        self._show_placeholder()
    
    def save_frame(self, filename: str) -> bool:
        """
        Save current frame to file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if saved successfully
        """
        try:
            pixmap = self.pixmap()
            if pixmap and not pixmap.isNull():
                return pixmap.save(filename)
            return False
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            return False