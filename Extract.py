import cv2
import pytesseract
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

class VideoTextExtractor:
    def __init__(self):
        # Cấu hình đường dẫn Tesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Thiết lập các đường dẫn
        self.video_dir = os.path.join('Dataset', 'Video')
        self.text_dir = os.path.join('Dataset', 'Text')
        self.output_file = os.path.join(self.text_dir, 'Label.csv')
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.text_dir, exist_ok=True)
        
        # Cấu hình OCR
        self.config = '--psm 7 -l vie'  # PSM 7 cho single line + Vietnamese language
        
    def extract_text_from_frame(self, frame):
        """Trích xuất text từ vùng góc phải trên của frame"""
        h, w = frame.shape[:2]
        
        # Xác định vùng ROI (góc phải trên)
        roi_x = int(w * 0.6)  # Lấy 40% bên phải
        roi_y = 0
        roi_w = w - roi_x
        roi_h = int(h * 0.15)  # Lấy 15% chiều cao từ trên xuống
        
        # Cắt vùng ROI
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Tiền xử lý ảnh
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Nhận dạng text
        text = pytesseract.image_to_string(thresh, config=self.config)
        return text.strip()
        
    def process_video(self, video_path):
        """Xử lý một video và trích xuất text"""
        cap = cv2.VideoCapture(video_path)
        
        # Đọc frame đầu tiên
        ret, frame = cap.read()
        if not ret:
            return None
            
        # Trích xuất text
        text = self.extract_text_from_frame(frame)
        
        cap.release()
        return text
        
    def process_all_videos(self):
        """Xử lý tất cả video trong thư mục"""
        print("Bắt đầu trích xuất text từ videos...")
        
        # Lấy danh sách video
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.mp4')]
        
        # Chuẩn bị data cho DataFrame
        data = []
        for idx, video_file in enumerate(tqdm(video_files, desc="Xử lý video"), 1):
            video_path = os.path.join(self.video_dir, video_file)
            text = self.process_video(video_path)
            
            if text:
                data.append({
                    'STT': idx,
                    'VIDEO': video_file,
                    'TEXT': text
                })
                
        # Tạo và lưu DataFrame
        df = pd.DataFrame(data)
        df.to_csv(self.output_file, index=False)
        
        print(f"\nĐã xử lý xong {len(data)} videos")
        print(f"Kết quả đã được lưu vào: {self.output_file}")
        
        # Hiển thị một số dòng đầu tiên
        print("\nMẫu dữ liệu:")
        print(df.head())

if __name__ == "__main__":
    extractor = VideoTextExtractor()
    extractor.process_all_videos()
