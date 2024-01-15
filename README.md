# Vietnamese Sign Language Detection System

> Cập nhật tài liệu: bổ sung mô tả pipeline và yêu cầu dữ liệu (README refresh)

### Pipeline dữ liệu nhanh
- Đặt video vào thư mục `Dataset/Video/` và nhãn vào `Dataset/Text/Label.csv` với cột `Video,Label`.
- Chạy xử lý: `python main.py --process-data` để tạo `Data/keypoints.npy` và `Data/labels.npy`.
- Huấn luyện: `python main.py --train` để huấn luyện mô hình và ghi kết quả vào `Models/`.

### Yêu cầu môi trường chính
- Python 3.10 (khuyến nghị)
- TensorFlow 2.15, MediaPipe 0.10.5, NumPy 1.26.x


> A Vietnamese Sign Language recognition system optimized for CPU and small datasets.

## 🚀 Features
- Lightweight Deep Learning models optimized for CPU  
- Real-time video processing (15–25 FPS)  
- Modern and intuitive PyQt5 interface with vertical sidebar  
- Smart data augmentation for limited datasets  
- Supports traditional ML, Deep Learning, and Ensemble methods  
- Modular architecture, detailed logging, and YAML configuration  

## 📁 Main Components
- **Core**: Detection and training engines  
- **Data**: Video processing, keypoints extraction, augmentation  
- **UI**: User interface for real-time visualization  
- **Utils**: Config manager, logging system  
- **Configs**: YAML configuration files  
- **Models**: Trained models storage  
- **Dataset**: Raw video data and labels  
- **Logs**: Application and training logs  

## 🛠️ Installation
Requirements: Python 3.8+, ≥8GB RAM, camera for real-time detection  

```bash
pip install -r requirements.txt
# or
pip install tensorflow-cpu opencv-python mediapipe PyQt5 scikit-learn
```

## 🚀 Usage
- **Run GUI application**:  
  `python main.py --gui`  

- **Train model**:  
  `python main.py --train --config custom.yaml`  

- **Process video dataset**:  
  `python main.py --process-data`  

## 📈 Demo Workflow
1. **Prepare Dataset**  
   - Place videos in `Dataset/Video/`  
   - Store labels in `Dataset/Text/Label.csv`  

2. **Process Data**  
   - Run `python main.py --process-data`  
   - Extract keypoints, augment data, save to `Data/`  

3. **Train Model**  
   - Run `python main.py --train`  
   - Perform cross-validation, select best model, save to `Models/`  

4. **Run Real-Time Detection**  
   - Run `python main.py --gui`  
   - Load trained model, open camera, recognize signs in real-time  
