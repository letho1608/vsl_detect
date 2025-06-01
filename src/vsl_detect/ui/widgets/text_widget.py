"""
Text display and control widget for Vietnamese Sign Language Detection.
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ...utils.config import Config
from ...utils.logger import get_logger


class TextWidget(QWidget):
    """Widget for displaying and controlling detected text."""
    
    # Signals
    speak_requested = pyqtSignal(str)
    clear_requested = pyqtSignal()
    
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.logger = get_logger(__name__)
        
        # Setup UI
        self._setup_ui()
        
        # Text state
        self.accumulated_text = []
    
    def _setup_ui(self) -> None:
        """Setup widget UI."""
        layout = QVBoxLayout(self)
        
        # Group box
        group = QGroupBox("Văn bản nhận dạng")
        group_layout = QVBoxLayout(group)
        
        # Text display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMinimumHeight(200)
        self.text_display.setMaximumHeight(400)
        
        # Styling
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
                line-height: 1.5;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        
        group_layout.addWidget(self.text_display)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        # Speak button
        self.speak_button = QPushButton("🔊 Đọc văn bản")
        self.speak_button.setToolTip("Đọc văn bản hiện tại")
        self.speak_button.clicked.connect(self._on_speak_clicked)
        button_layout.addWidget(self.speak_button)
        
        # Clear button
        self.clear_button = QPushButton("🗑️ Xóa văn bản")
        self.clear_button.setToolTip("Xóa tất cả văn bản")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        button_layout.addWidget(self.clear_button)
        
        # Copy button
        self.copy_button = QPushButton("📋 Sao chép")
        self.copy_button.setToolTip("Sao chép văn bản vào clipboard")
        self.copy_button.clicked.connect(self._on_copy_clicked)
        button_layout.addWidget(self.copy_button)
        
        # Save button
        self.save_button = QPushButton("💾 Lưu")
        self.save_button.setToolTip("Lưu văn bản vào file")
        self.save_button.clicked.connect(self._on_save_clicked)
        button_layout.addWidget(self.save_button)
        
        group_layout.addLayout(button_layout)
        
        # Statistics
        stats_layout = QHBoxLayout()
        
        self.word_count_label = QLabel("Từ: 0")
        self.char_count_label = QLabel("Ký tự: 0")
        
        stats_layout.addWidget(self.word_count_label)
        stats_layout.addWidget(self.char_count_label)
        stats_layout.addStretch()
        
        group_layout.addLayout(stats_layout)
        
        layout.addWidget(group)
        
        # Style buttons
        button_style = """
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f9fa;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
            }
        """
        
        for button in [self.speak_button, self.clear_button, self.copy_button, self.save_button]:
            button.setStyleSheet(button_style)
    
    def add_text(self, text: str) -> None:
        """
        Add new detected text.
        
        Args:
            text: Detected text to add
        """
        if not text.strip():
            return
        
        # Add to accumulated text
        self.accumulated_text.append(text.strip())
        
        # Update display
        self._update_display()
        
        # Update statistics
        self._update_statistics()
        
        # Auto-scroll to bottom
        self.text_display.verticalScrollBar().setValue(
            self.text_display.verticalScrollBar().maximum()
        )
        
        self.logger.debug(f"Added text: {text}")
    
    def _update_display(self) -> None:
        """Update text display."""
        # Join words with spaces
        full_text = " ".join(self.accumulated_text)
        
        # Set text with basic formatting
        formatted_text = f"<p style='margin: 0; padding: 0;'>{full_text}</p>"
        self.text_display.setHtml(formatted_text)
    
    def _update_statistics(self) -> None:
        """Update word and character statistics."""
        full_text = " ".join(self.accumulated_text)
        
        word_count = len([word for word in full_text.split() if word.strip()])
        char_count = len(full_text)
        
        self.word_count_label.setText(f"Từ: {word_count}")
        self.char_count_label.setText(f"Ký tự: {char_count}")
    
    def get_text(self) -> str:
        """Get current accumulated text."""
        return " ".join(self.accumulated_text)
    
    def clear_text(self) -> None:
        """Clear all text."""
        self.accumulated_text.clear()
        self.text_display.clear()
        self._update_statistics()
        self.logger.debug("Text cleared")
    
    def _on_speak_clicked(self) -> None:
        """Handle speak button click."""
        text = self.get_text()
        if text:
            self.speak_requested.emit(text)
        else:
            QMessageBox.information(self, "Thông báo", "Không có văn bản để đọc.")
    
    def _on_clear_clicked(self) -> None:
        """Handle clear button click."""
        if self.accumulated_text:
            reply = QMessageBox.question(
                self,
                "Xác nhận",
                "Bạn có chắc chắn muốn xóa tất cả văn bản?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.clear_text()
                self.clear_requested.emit()
    
    def _on_copy_clicked(self) -> None:
        """Handle copy button click."""
        text = self.get_text()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            QMessageBox.information(self, "Thông báo", "Đã sao chép văn bản vào clipboard.")
        else:
            QMessageBox.information(self, "Thông báo", "Không có văn bản để sao chép.")
    
    def _on_save_clicked(self) -> None:
        """Handle save button click."""
        text = self.get_text()
        if not text:
            QMessageBox.information(self, "Thông báo", "Không có văn bản để lưu.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu văn bản",
            "nhan_dang_ky_hieu.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                QMessageBox.information(self, "Thành công", f"Đã lưu văn bản vào:\n{filename}")
                self.logger.info(f"Text saved to: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu file:\n{str(e)}")
                self.logger.error(f"Failed to save text: {e}")
    
    def set_font_size(self, size: int) -> None:
        """Set text display font size."""
        current_style = self.text_display.styleSheet()
        new_style = current_style.replace(
            f"font-size: {16}px",
            f"font-size: {size}px"
        )
        self.text_display.setStyleSheet(new_style)
    
    def export_html(self, filename: str) -> bool:
        """
        Export text as HTML file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if exported successfully
        """
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Kết quả nhận dạng ngôn ngữ ký hiệu</title>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; }}
                    .header {{ color: #333; border-bottom: 2px solid #ccc; padding-bottom: 10px; }}
                    .content {{ margin-top: 20px; line-height: 1.6; }}
                    .stats {{ margin-top: 20px; color: #666; font-size: 14px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Kết quả nhận dạng ngôn ngữ ký hiệu</h1>
                </div>
                <div class="content">
                    <p>{self.get_text()}</p>
                </div>
                <div class="stats">
                    <p>Số từ: {len(self.get_text().split())}</p>
                    <p>Số ký tự: {len(self.get_text())}</p>
                    <p>Xuất lúc: {QDateTime.currentDateTime().toString()}</p>
                </div>
            </body>
            </html>
            """
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export HTML: {e}")
            return False